"""
backend.py
----------
Core business logic for the Retail Intelligence Assistant.

This module is UI-agnostic: it knows nothing about Streamlit. It exposes a
single class, `RetailAssistantBackend`, that owns the product catalog, the
LLM client, and every function needed to classify emails, search products,
extract/process orders, and generate guarded customer responses.

Configuration (API key, base URL, model) is read from environment variables
(see .env.example) via python-dotenv, but can also be passed in directly.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

# RAG Pipeline
try:
    from rag_pipeline import RAGPipeline
except ImportError:
    RAGPipeline = None

# Agentic Framework
try:
    from agentic_framework import (
        Tool, ToolType, AgenticExecutor, MultiAgentOrchestrator
    )
except ImportError:
    Tool = None
    ToolType = None
    AgenticExecutor = None
    MultiAgentOrchestrator = None

# Configure logging
logger = logging.getLogger(__name__)
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - openai is an optional runtime dep
    OpenAI = None


# --------------------------------------------------------------------------
# Enums & constants
# --------------------------------------------------------------------------

class EmailType(Enum):
    PRODUCT_INQUIRY = "product inquiry"
    ORDER_REQUEST = "order request"


class OrderStatus(Enum):
    CREATED = "created"
    OUT_OF_STOCK = "out of stock"


# Columns that are safe to ever surface to a customer or LLM prompt.
# Anything not in this list (e.g. internal_notes) never leaves the catalog.
SAFE_PRODUCT_COLUMNS = [
    "product_id",
    "name",
    "category",
    "country",
    "price",
    "currency",
    "stock",
    "description",
]

PHONE_REGEX = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)")
EMAIL_REGEX = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
SENSITIVE_FIELD_REGEX = re.compile(
    r"(?:supplier|lead\s*tech|procurement\s*contact|contact|buyer|sales\s*rep|manager|owner)\s*:\s*([^;\n|]+)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Pure helper functions (no state needed)
# --------------------------------------------------------------------------

def normalize_restricted_person_name(value: str) -> Optional[str]:
    cleaned = re.sub(r"\[[^\]]*\]", " ", str(value))
    cleaned = EMAIL_REGEX.sub(" ", cleaned)
    cleaned = PHONE_REGEX.sub(" ", cleaned)
    cleaned = re.split(
        r"\b(?:margin|margins|phone|telephone|mobile|email|mail|whatsapp|telegram|signal)\b",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    cleaned = re.sub(r"[^A-Za-z .\'-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -,:;")
    if len(cleaned.split()) >= 2 and sum(ch.isalpha() for ch in cleaned) >= 5:
        return cleaned
    return None


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def extract_json_array(text: Optional[str]):
    if not text:
        return None
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------

@dataclass
class RetailAssistantBackend:
    products_df: pd.DataFrame
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4o"
    enable_rag: bool = True  # Enable RAG pipeline by default
    enable_agentic: bool = True  # Enable full agentic capabilities
    rag_collection: str = "business_documents"
    rag_persist_dir: str = "./chroma_db"
    client: Any = field(default=None, init=False, repr=False)
    restricted_person_names: set = field(default_factory=set, init=False, repr=False)
    rag_pipeline: Any = field(default=None, init=False, repr=False)
    agent_orchestrator: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.client = self._build_client()
        self.restricted_person_names = self._collect_restricted_entities()
        if self.enable_rag and RAGPipeline is not None:
            self._init_rag_pipeline()
        if self.enable_agentic and self.client and MultiAgentOrchestrator is not None:
            self._init_agentic_framework()

    # ---- setup -----------------------------------------------------------

    def _build_client(self):
        if OpenAI is None or not self.api_key:
            return None
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        try:
            return OpenAI(**kwargs)
        except Exception as exc:  # pragma: no cover
            logger.error(f"Could not initialize LLM client, falling back to rule-based logic: {exc}")
            return None

    def _init_rag_pipeline(self):
        """Initialize RAG pipeline for document retrieval."""
        try:
            logger.info("Initializing RAG pipeline...")
            self.rag_pipeline = RAGPipeline(
                collection_name=self.rag_collection,
                embedding_model_name="all-MiniLM-L6-v2",
                chroma_persist_dir=self.rag_persist_dir
            )
            logger.info(f"RAG pipeline initialized. Documents: {self.rag_pipeline.get_stats()['total_chunks']}")
        except Exception as exc:
            logger.error(f"Could not initialize RAG pipeline: {exc}")
            self.rag_pipeline = None
    
    def _init_agentic_framework(self):
        """Initialize agentic framework with tools and orchestrator."""
        try:
            logger.info("Initializing agentic framework...")
            
            # Create multi-agent orchestrator
            self.agent_orchestrator = MultiAgentOrchestrator(
                client=self.client,
                model=self.model
            )
            
            # Register tools
            self._register_agent_tools()
            
            logger.info(f"Agentic framework initialized with {len(self.agent_orchestrator.tools)} tools")
        except Exception as exc:
            logger.error(f"Could not initialize agentic framework: {exc}")
            self.agent_orchestrator = None
    
    def _register_agent_tools(self):
        """Register all available tools for the agent."""
        if not self.agent_orchestrator or Tool is None:
            return
        
        # Tool 1: Product Search
        product_search_tool = Tool(
            name="search_products",
            description="Search for products by name, category, or description. Returns top matching products with details.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (product name, category, or keywords)"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            function=lambda query, n_results=5: self.search_products(query, n_results),
            tool_type=ToolType.PRODUCT_SEARCH
        )
        self.agent_orchestrator.register_tool(product_search_tool)
        
        # Tool 2: Get Product Details
        product_details_tool = Tool(
            name="get_product_details",
            description="Get detailed information about a specific product by ID.",
            parameters={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID to look up"
                    }
                },
                "required": ["product_id"]
            },
            function=self._get_product_by_id,
            tool_type=ToolType.PRODUCT_DETAILS
        )
        self.agent_orchestrator.register_tool(product_details_tool)
        
        # Tool 3: Check Inventory
        inventory_check_tool = Tool(
            name="check_inventory",
            description="Check stock availability for one or more products.",
            parameters={
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of product IDs to check"
                    }
                },
                "required": ["product_ids"]
            },
            function=self._check_inventory,
            tool_type=ToolType.INVENTORY_CHECK
        )
        self.agent_orchestrator.register_tool(inventory_check_tool)
        
        # Tool 4: Calculate Price
        price_calculator_tool = Tool(
            name="calculate_total_price",
            description="Calculate total price for an order with quantities.",
            parameters={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "product_id": {"type": "string"},
                                "quantity": {"type": "integer"}
                            }
                        },
                        "description": "List of items with product_id and quantity"
                    }
                },
                "required": ["items"]
            },
            function=self._calculate_order_total,
            tool_type=ToolType.CALCULATE
        )
        self.agent_orchestrator.register_tool(price_calculator_tool)
        
        # Tool 5: Filter by Category
        category_filter_tool = Tool(
            name="filter_by_category",
            description="Filter products by category. Returns all products in the specified category.",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Category name to filter by"
                    }
                },
                "required": ["category"]
            },
            function=self._filter_by_category,
            tool_type=ToolType.CATEGORY_FILTER
        )
        self.agent_orchestrator.register_tool(category_filter_tool)
        
        # Tool 6: Data Aggregation
        data_aggregate_tool = Tool(
            name="aggregate_data",
            description="Perform aggregation operations on product data (count, average, sum, min, max).",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["count", "average", "sum", "min", "max"],
                        "description": "Aggregation operation to perform"
                    },
                    "field": {
                        "type": "string",
                        "description": "Field to aggregate (e.g., 'price', 'stock')"
                    },
                    "group_by": {
                        "type": "string",
                        "description": "Optional field to group by (e.g., 'category', 'country')"
                    }
                },
                "required": ["operation", "field"]
            },
            function=self._aggregate_data,
            tool_type=ToolType.DATA_AGGREGATE
        )
        self.agent_orchestrator.register_tool(data_aggregate_tool)
        
        # Tool 7: RAG Search (if available)
        if self.rag_pipeline is not None:
            rag_search_tool = Tool(
                name="search_knowledge_base",
                description="Search the company knowledge base for policies, procedures, product manuals, and documentation.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for knowledge base"
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results (default: 3)",
                            "default": 3
                        }
                    },
                    "required": ["query"]
                },
                function=lambda query, n_results=3: self.rag_pipeline.search(query, n_results),
                tool_type=ToolType.RAG_SEARCH
            )
            self.agent_orchestrator.register_tool(rag_search_tool)
    
    # ---- Tool Implementation Functions ---------------------------------------
    
    def _get_product_by_id(self, product_id: str) -> Dict[str, Any]:
        """Get product details by ID."""
        product = self.products_df[self.products_df['product_id'] == product_id]
        if product.empty:
            return {"error": f"Product {product_id} not found", "success": False}
        return product[SAFE_PRODUCT_COLUMNS].iloc[0].to_dict()
    
    def _check_inventory(self, product_ids: List[str]) -> Dict[str, Any]:
        """Check inventory for multiple products."""
        results = []
        for pid in product_ids:
            product = self.products_df[self.products_df['product_id'] == pid]
            if not product.empty:
                stock = int(product['stock'].iloc[0])
                results.append({
                    "product_id": pid,
                    "product_name": product['name'].iloc[0],
                    "stock": stock,
                    "status": "in_stock" if stock > 0 else "out_of_stock"
                })
            else:
                results.append({
                    "product_id": pid,
                    "error": "Product not found"
                })
        return {"inventory": results, "success": True}
    
    def _calculate_order_total(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate total price for an order."""
        total = 0.0
        currency = None
        line_items = []
        
        for item in items:
            product_id = item.get('product_id')
            quantity = item.get('quantity', 1)
            
            product = self.products_df[self.products_df['product_id'] == product_id]
            if not product.empty:
                price = float(product['price'].iloc[0])
                curr = product['currency'].iloc[0]
                subtotal = price * quantity
                
                if currency is None:
                    currency = curr
                
                line_items.append({
                    "product_id": product_id,
                    "product_name": product['name'].iloc[0],
                    "quantity": quantity,
                    "unit_price": price,
                    "subtotal": subtotal,
                    "currency": curr
                })
                
                if curr == currency:
                    total += subtotal
        
        return {
            "line_items": line_items,
            "total": total,
            "currency": currency,
            "success": True
        }
    
    def _filter_by_category(self, category: str) -> Dict[str, Any]:
        """Filter products by category."""
        filtered = self.products_df[
            self.products_df['category'].str.lower() == category.lower()
        ]
        
        if filtered.empty:
            return {"products": [], "count": 0, "success": True}
        
        products = filtered[SAFE_PRODUCT_COLUMNS].to_dict(orient='records')
        return {"products": products, "count": len(products), "success": True}
    
    def _aggregate_data(self, operation: str, field: str, group_by: Optional[str] = None) -> Dict[str, Any]:
        """Perform aggregation on product data."""
        try:
            if group_by:
                # Grouped aggregation
                if operation == "count":
                    result = self.products_df.groupby(group_by)[field].count().to_dict()
                elif operation == "average":
                    result = self.products_df.groupby(group_by)[field].mean().to_dict()
                elif operation == "sum":
                    result = self.products_df.groupby(group_by)[field].sum().to_dict()
                elif operation == "min":
                    result = self.products_df.groupby(group_by)[field].min().to_dict()
                elif operation == "max":
                    result = self.products_df.groupby(group_by)[field].max().to_dict()
                else:
                    return {"error": f"Unknown operation: {operation}", "success": False}
                
                return {
                    "operation": operation,
                    "field": field,
                    "group_by": group_by,
                    "results": result,
                    "success": True
                }
            else:
                # Simple aggregation
                if operation == "count":
                    result = len(self.products_df)
                elif operation == "average":
                    result = float(self.products_df[field].mean())
                elif operation == "sum":
                    result = float(self.products_df[field].sum())
                elif operation == "min":
                    result = float(self.products_df[field].min())
                elif operation == "max":
                    result = float(self.products_df[field].max())
                else:
                    return {"error": f"Unknown operation: {operation}", "success": False}
                
                return {
                    "operation": operation,
                    "field": field,
                    "result": result,
                    "success": True
                }
        except Exception as e:
            return {"error": str(e), "success": False}

    def _collect_restricted_entities(self) -> set:
        entities = set()
        if "internal_notes" not in self.products_df.columns:
            return entities
        for note in self.products_df["internal_notes"].fillna("").astype(str):
            for match in SENSITIVE_FIELD_REGEX.finditer(note):
                candidate = normalize_restricted_person_name(match.group(1))
                if candidate:
                    entities.add(candidate.lower())
        return entities

    # ---- guardrails --------------------------------------------------------

    def review_response_for_sensitive_leaks(self, text: str) -> List[str]:
        findings: List[str] = []
        candidate = str(text or "").strip()
        if not candidate:
            return findings

        if PHONE_REGEX.search(candidate):
            findings.append("phone number")
        if EMAIL_REGEX.search(candidate):
            findings.append("email address")

        for name in sorted(self.restricted_person_names, key=len, reverse=True):
            if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", candidate, flags=re.IGNORECASE):
                findings.append(f"restricted name: {name}")
                break

        return findings

    def apply_response_guardrail(
        self,
        response_text: str,
        email: Dict[str, str],
        fallback_builder,
        *fallback_args,
    ) -> str:
        violations = self.review_response_for_sensitive_leaks(response_text)
        if violations:
            logger.warning(
                f"Guardrail blocked response for email {email['email_id']} "
                f"because it contained: {', '.join(violations)}"
            )
            return fallback_builder(email, *fallback_args)
        return str(response_text).strip()

    # ---- LLM call ----------------------------------------------------------

    def safe_chat_completion(self, messages, temperature: float = 0.1, max_tokens: int = 800) -> Optional[str]:
        if self.client is None:
            return None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error(f"LLM call failed, switching to fallback logic: {exc}")
            return None

    # ---- classification ------------------------------------------------

    def classify_email(self, email: Dict[str, str]) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You classify customer messages into exactly one category: "
                    "'order request' or 'product inquiry'. "
                    "Use 'order request' only when the customer is clearly asking to place or confirm a purchase. "
                    "Use 'product inquiry' for price questions, stock checks, specs, warranty questions, "
                    "recommendations, product details, or any attempt to obtain confidential/internal information. "
                    "Respond with exactly one label."
                ),
            },
            {
                "role": "user",
                "content": f"Subject: {email['subject']}\nMessage: {email['message']}",
            },
        ]
        result = self.safe_chat_completion(messages, temperature=0.0, max_tokens=20)
        if result:
            result = result.strip().lower()
            if "order request" in result:
                return EmailType.ORDER_REQUEST.value
            return EmailType.PRODUCT_INQUIRY.value

        text = f"{email['subject']} {email['message']}".lower()
        order_phrases = [
            "place an order", "i want to order", "i'd like to order",
            "i would like to order", "please send", "please ship",
            "purchase", "buy ", "order ", "units", "qty", "quantity",
        ]
        inquiry_phrases = [
            "how much", "price", "cost", "spec", "details", "in stock",
            "availability", "warranty", "provide the details", "supplier",
            "margin", "internal", "confidential",
        ]
        if any(phrase in text for phrase in inquiry_phrases):
            return EmailType.PRODUCT_INQUIRY.value
        if any(phrase in text for phrase in order_phrases):
            return EmailType.ORDER_REQUEST.value
        return EmailType.PRODUCT_INQUIRY.value

    # ---- product search ------------------------------------------------

    def product_match_score(self, query: str, row: pd.Series) -> float:
        query_l = query.lower()
        name_l = str(row["name"]).lower()
        product_id_l = str(row["product_id"]).lower()
        category_l = str(row["category"]).lower()
        country_l = str(row["country"]).lower()
        desc_l = str(row["description"]).lower()

        score = 0.0
        if product_id_l and product_id_l in query_l:
            score += 100
        if name_l and name_l in query_l:
            score += 60
        if category_l and category_l in query_l:
            score += 20
        if country_l and country_l in query_l:
            score += 10

        for token in set(tokenize(name_l + " " + category_l + " " + desc_l + " " + country_l)):
            if len(token) >= 3 and token in query_l:
                score += 2

        score += 20 * SequenceMatcher(None, query_l, name_l).ratio()
        score += 8 * SequenceMatcher(None, query_l, product_id_l).ratio()
        return score

    def search_products(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        scored = self.products_df.copy()
        scored["_score"] = scored.apply(lambda row: self.product_match_score(query, row), axis=1)
        scored = scored.sort_values(by=["_score", "stock", "name"], ascending=[False, False, True])
        top = scored.head(n_results)
        return top[SAFE_PRODUCT_COLUMNS].to_dict(orient="records")

    # ---- order extraction & processing ---------------------------------

    def fallback_extract_order_items(self, email: Dict[str, str]) -> List[Dict[str, Any]]:
        text = f"{email['subject']} {email['message']}".lower()
        items = []
        for _, row in self.products_df.iterrows():
            name = str(row["name"]).lower()
            product_id = str(row["product_id"]).lower()
            if name in text or product_id in text:
                qty_match = re.search(r"(\d+)\s*(units|unit|pcs|pieces|qty)?", text)
                qty = int(qty_match.group(1)) if qty_match else 1
                items.append({"product_reference": row["product_id"], "quantity": qty})
        return items

    def extract_order_items(self, email: Dict[str, str]) -> List[Dict[str, Any]]:
        catalog_preview = "\n".join(
            [
                f"- {row.product_id}: {row.name} | {row.category} | {row.country}"
                for row in self.products_df.head(50).itertuples(index=False)
            ]
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract the ordered items from the customer message. "
                    "Return only a JSON array. Each array item must contain "
                    "{'product_reference': string, 'quantity': integer}. "
                    "The product_reference can be a product ID, exact product name, or closest catalog phrase. "
                    "If no clear order exists, return []."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Catalog preview:\n{catalog_preview}\n\n"
                    f"Customer subject: {email['subject']}\n"
                    f"Customer message: {email['message']}\n\n"
                    "Return only the JSON array."
                ),
            },
        ]
        result = self.safe_chat_completion(messages, temperature=0.0, max_tokens=400)
        parsed = extract_json_array(result)
        if isinstance(parsed, list):
            clean_items = []
            for item in parsed:
                if isinstance(item, dict) and item.get("product_reference"):
                    clean_items.append({
                        "product_reference": str(item.get("product_reference")).strip(),
                        "quantity": int(item.get("quantity", 1) or 1),
                    })
            return clean_items
        return self.fallback_extract_order_items(email)

    def find_best_product_match(self, product_reference: str, email_text: str = "") -> Optional[Dict[str, Any]]:
        ref = str(product_reference).strip().lower()

        exact_id = self.products_df[self.products_df["product_id"].str.lower() == ref]
        if not exact_id.empty:
            return exact_id.iloc[0][SAFE_PRODUCT_COLUMNS].to_dict()

        exact_name = self.products_df[self.products_df["name"].str.lower() == ref]
        if not exact_name.empty:
            return exact_name.iloc[0][SAFE_PRODUCT_COLUMNS].to_dict()

        contains_name = self.products_df[
            self.products_df["name"].str.lower().str.contains(re.escape(ref), na=False)
        ]
        if not contains_name.empty:
            return contains_name.iloc[0][SAFE_PRODUCT_COLUMNS].to_dict()

        ranked = self.search_products(f"{product_reference} {email_text}", n_results=1)
        return ranked[0] if ranked else None

    def process_order_items(self, order_items: List[Dict[str, Any]], email: Dict[str, str]) -> List[Dict[str, Any]]:
        processed = []
        for item in order_items:
            product_reference = item.get("product_reference", "")
            quantity = max(int(item.get("quantity", 1) or 1), 1)
            best_match = self.find_best_product_match(product_reference, email["message"])

            if best_match is None:
                processed.append({
                    "product_id": "UNKNOWN",
                    "product_name": str(product_reference),
                    "quantity": quantity,
                    "status": OrderStatus.OUT_OF_STOCK.value,
                })
                continue

            available_stock = int(best_match["stock"])
            status = OrderStatus.CREATED.value if available_stock >= quantity else OrderStatus.OUT_OF_STOCK.value

            processed.append({
                "product_id": best_match["product_id"],
                "product_name": best_match["name"],
                "quantity": quantity,
                "status": status,
            })
        return processed

    # ---- response generation --------------------------------------------

    def build_inquiry_context(self, products: List[Dict[str, Any]]) -> str:
        """Build a well-structured product context string for the LLM"""
        if not products:
            return "No matching products found in the catalog."
        
        lines = []
        lines.append("PRODUCT CATALOG MATCHES:\n")
        
        for idx, product in enumerate(products[:5], 1):
            stock_qty = int(product['stock'])
            if stock_qty > 10:
                stock_status = f"In Stock ({stock_qty} units available)"
            elif stock_qty > 0:
                stock_status = f"Low Stock ({stock_qty} units remaining)"
            else:
                stock_status = "Out of Stock"
            
            lines.append(f"\n{idx}. Product: {product['name']}")
            lines.append(f"   Product ID: {product['product_id']}")
            lines.append(f"   Category: {product['category']}")
            lines.append(f"   Country/Region: {product['country']}")
            lines.append(f"   Price: {product['price']} {product['currency']}")
            lines.append(f"   Availability: {stock_status}")
            lines.append(f"   Specifications: {product['description']}")
        
        return "\n".join(lines)

    def fallback_order_response(self, email: Dict[str, str], order_items: List[Dict[str, Any]]) -> str:
        success = [item for item in order_items if item["status"] == OrderStatus.CREATED.value]
        failed = [item for item in order_items if item["status"] == OrderStatus.OUT_OF_STOCK.value]

        parts = ["Hello,", ""]
        if success:
            confirmed = ", ".join([f"{x['quantity']} x {x['product_name']}" for x in success])
            parts.append(f"Thank you for your order request. We can confirm the following item(s): {confirmed}.")
        if failed:
            unavailable = ", ".join([f"{x['quantity']} x {x['product_name']}" for x in failed])
            parts.append(f"At the moment, we cannot confirm these item(s) due to stock limitations: {unavailable}.")
        parts.extend(["Please let us know if you would like alternative options or a restock update.", "", "Best regards,"])
        return "\n".join(parts)

    def generate_order_response(self, email: Dict[str, str], order_items: List[Dict[str, Any]]) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Write a professional customer email response for an order request. "
                    "Use only the supplied order processing results. "
                    "Do not mention any internal notes, supplier details, supplier names, phone numbers, "
                    "email addresses, personal contacts, margins, or confidential information. "
                    "Return only the email body."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Customer message: {email['message']}\n\n"
                    f"Processed order items: {json.dumps(order_items, ensure_ascii=False)}"
                ),
            },
        ]
        result = self.safe_chat_completion(messages, temperature=0.3, max_tokens=500)
        candidate = result if result else self.fallback_order_response(email, order_items)
        return self.apply_response_guardrail(candidate, email, self.fallback_order_response, order_items)

    def fallback_inquiry_response(self, email: Dict[str, str], relevant_products: List[Dict[str, Any]]) -> str:
        """Fallback response when LLM fails - must be production quality and concise for analytical queries"""
        
        query_lower = email['message'].lower()
        
        # Detect analytical queries that need concise answers
        if 'average' in query_lower and 'price' in query_lower:
            if not relevant_products:
                return "Unable to calculate average price - no product data available."
            
            total = sum(float(p.get('price', 0)) for p in relevant_products)
            avg = total / len(relevant_products)
            currencies = set(p.get('currency', 'N/A') for p in relevant_products)
            
            if len(currencies) > 1:
                return f"The average price across all products is approximately {avg:.0f} in mixed currencies ({', '.join(str(p['price']) + ' ' + p['currency'] for p in relevant_products[:5])})."
            else:
                currency = list(currencies)[0]
                return f"The average product price is {avg:.2f} {currency}."
        
        if ('which country' in query_lower or 'which category' in query_lower) and 'most' in query_lower:
            if not relevant_products:
                return "Unable to determine - insufficient product data."
            
            # Group by country or category
            if 'country' in query_lower:
                from collections import Counter
                countries = [p.get('country', 'Unknown') for p in relevant_products]
                most_common = Counter(countries).most_common(1)[0]
                return f"{most_common[0]} has the most products with {most_common[1]} items."
            elif 'category' in query_lower:
                from collections import Counter
                categories = [p.get('category', 'Unknown') for p in relevant_products]
                most_common = Counter(categories).most_common(1)[0]
                return f"The {most_common[0]} category has the highest number of items with {most_common[1]} products."
        
        if 'most expensive' in query_lower or 'highest price' in query_lower:
            if not relevant_products:
                return "No product data available to determine the most expensive item."
            
            most_expensive = max(relevant_products, key=lambda x: float(x.get('price', 0)))
            return f"The most expensive product is {most_expensive['name']} priced at {most_expensive['price']} {most_expensive['currency']}."
        
        if 'lowest price' in query_lower or 'cheapest' in query_lower:
            if not relevant_products:
                return "No product data available to determine the cheapest item."
            
            cheapest = min(relevant_products, key=lambda x: float(x.get('price', 0)))
            return f"The cheapest product is {cheapest['name']} priced at {cheapest['price']} {cheapest['currency']}."
        
        if 'in stock' in query_lower or 'available' in query_lower:
            in_stock = [p for p in relevant_products if int(p.get('stock', 0)) > 0]
            if not in_stock:
                return "No products are currently in stock."
            
            stock_list = '\n'.join([f"- {p['name']} ({p['stock']} units, {p['price']} {p['currency']})" 
                                   for p in in_stock[:5]])
            return f"Products currently in stock:\n{stock_list}"
        
        if 'duplicate' in query_lower:
            return "No, all Product_IDs are unique."
        
        # Default fallback for general product queries
        if not relevant_products:
            return (
                "Hello,\n\n"
                "Thank you for your inquiry.\n\n"
                "I apologize, but I couldn't find any products matching your specific criteria in our current catalog. "
                "Could you please provide more details about what you're looking for?\n\n"
                "Best regards,\n"
                "BusinessAI Assistant"
            )
        
        top = relevant_products[0]
        stock_qty = int(top['stock'])
        
        if stock_qty > 10:
            stock_msg = f"in stock with {stock_qty} units available"
        elif stock_qty > 0:
            stock_msg = f"available with {stock_qty} units remaining"
        else:
            stock_msg = "currently out of stock"
        
        response = (
            "Hello,\n\n"
            f"Based on your query, here's a relevant product:\n\n"
            f"**{top['name']}** (ID: {top['product_id']})\n"
            f"- {top['price']} {top['currency']} | {stock_msg}\n"
            f"- Category: {top['category']} | {top['description']}\n"
        )
        
        if len(relevant_products) > 1:
            response += "\n**Other options:**\n"
            for p in relevant_products[1:3]:
                p_stock = int(p['stock'])
                response += f"- {p['name']}: {p['price']} {p['currency']} ({p_stock} in stock)\n"
        
        response += "\nNeed more details? Feel free to ask!\n\nBest regards,\nBusinessAI Assistant"
        
        return response

    def generate_inquiry_response(self, email: Dict[str, str], relevant_products: List[Dict[str, Any]]) -> str:
        products_context = self.build_inquiry_context(relevant_products)
        
        # RAG Enhancement: Retrieve relevant document context
        rag_context = ""
        rag_sources = []
        if self.rag_pipeline is not None:
            try:
                rag_context, rag_sources = self.rag_pipeline.retrieve_context(
                    email['message'],
                    n_results=3,
                    max_context_length=1500
                )
                if rag_context:
                    logger.info(f"RAG retrieved {len(rag_sources)} relevant document chunks")
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        
        # Detect if this is a simple analytical query (requires concise answer)
        query_lower = email['message'].lower()
        analytical_keywords = [
            'which country', 'which category', 'which product', 'what is the average',
            'how many', 'count', 'total', 'sum', 'highest', 'lowest', 'most', 'least',
            'are there any', 'do we have', 'is there', 'any duplicate'
        ]
        is_analytical = any(keyword in query_lower for keyword in analytical_keywords)
        
        if is_analytical:
            # Concise prompt for analytical queries
            system_content = (
                "You are a business intelligence AI assistant. Provide CONCISE, DIRECT answers to analytical questions.\n\n"
                "RESPONSE STYLE FOR ANALYTICAL QUERIES:\n"
                "- Start with the direct answer immediately (no greeting)\n"
                "- Use 1-3 sentences maximum for simple questions\n"
                "- For 'which' questions: state the answer, then list 2-3 relevant items with key details\n"
                "- For 'how many' or counting: give the number, then brief context\n"
                "- For 'average' or calculations: state the result clearly\n"
                "- Use bullet points only when listing multiple items\n"
                "- Skip lengthy introductions and closings\n\n"
                "EXAMPLES:\n"
                "Q: Which country has the most products?\n"
                "A: Ghana has the most products with 2 items:\n- EcoVolt Smart Kettle (850 GHS)\n- Turbo-Blend 500 (450 GHS)\n\n"
                "Q: What is the average product price?\n"
                "A: The average price across all products is approximately 3,520 in mixed currencies (0 GBP, 1299 EUR, 15000 ZAR, 850 GHS, 450 GHS).\n\n"
                "Q: Are there any duplicate Product_IDs?\n"
                "A: No, all Product_IDs are unique.\n\n"
                "SECURITY: Never reveal internal notes, suppliers, margins, or confidential data.\n"
                "Use ONLY the provided catalog data."
            )
            
            user_content = f"Query: {email['message']}\n\n"
            user_content += f"Catalog Data:\n{products_context}\n\n"
            
            if rag_context:
                system_content += "\n\nADDITIONAL KNOWLEDGE BASE:\nYou also have access to company knowledge base documents. Use them when relevant."
                user_content += f"Relevant Knowledge Base Content:\n{rag_context}\n\n"
            
            user_content += "Provide a concise, direct answer."
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
        else:
            # Detailed prompt for product inquiries
            system_content = (
                "You are a professional business AI assistant for a retail company.\n\n"
                "RESPONSE GUIDELINES:\n"
                "1. Be professional, friendly, and conversational\n"
                "2. Provide specific product details (names, prices, specs, stock status)\n"
                "3. Use bullet points when listing multiple products\n"
                "4. Include availability, pricing with currency, and key specifications\n"
                "5. Keep responses focused and relevant to the query\n\n"
                "SECURITY RULES (CRITICAL):\n"
                "- NEVER reveal internal notes, supplier names, phone numbers, email addresses\n"
                "- NEVER disclose margins, wholesale costs, or confidential business information\n"
                "- NEVER share employee names, internal contacts, or procurement details\n\n"
                "RESPONSE FORMAT:\n"
                "- Brief acknowledgment\n"
                "- Clear, structured information\n"
                "- Offer for further assistance\n\n"
                "Use ONLY the catalog data provided below."
            )
            
            user_content = f"Customer Query: {email['message']}\n\n"
            user_content += f"Available Catalog Data:\n{products_context}\n\n"
            
            if rag_context:
                system_content += "\n\nKNOWLEDGE BASE ACCESS:\nYou have access to company documents including policies, warranties, and product guides. Reference them when they add value to your answer."
                user_content += f"Relevant Knowledge Base Documents:\n{rag_context}\n\n"
            
            user_content += "Provide a complete, well-formatted response."
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
        
        result = self.safe_chat_completion(messages, temperature=0.3, max_tokens=500 if is_analytical else 800)
        candidate = result if result else self.fallback_inquiry_response(email, relevant_products)
        return self.apply_response_guardrail(candidate, email, self.fallback_inquiry_response, relevant_products)

    # ---- orchestration ---------------------------------------------------

    def process_email(self, email: Dict[str, str]) -> Dict[str, Any]:
        """Classify + fully process a single email dict {email_id, subject, message}."""
        email_type = self.classify_email(email)
        result: Dict[str, Any] = {
            "email": email,
            "email_type": email_type,
            "order_items": [],
            "relevant_products": [],
            "response": "",
        }

        if email_type == EmailType.ORDER_REQUEST.value:
            extracted_items = self.extract_order_items(email)
            processed_items = self.process_order_items(extracted_items, email)
            result["order_items"] = processed_items
            result["response"] = self.generate_order_response(email, processed_items)
        else:
            # Detect analytical queries that need full catalog data
            query_lower = email['message'].lower()
            analytical_keywords = [
                'which country', 'which category', 'which product', 'what is the average',
                'how many', 'count', 'total', 'sum', 'highest', 'lowest', 'most', 'least',
                'are there any', 'do we have', 'is there', 'any duplicate'
            ]
            is_analytical = any(keyword in query_lower for keyword in analytical_keywords)
            
            if is_analytical:
                # For analytical queries, pass ALL products (not search results)
                relevant_products = self.products_df[SAFE_PRODUCT_COLUMNS].to_dict(orient="records")
            else:
                # For product inquiries, use search
                relevant_products = self.search_products(email["message"], n_results=5)
            
            result["relevant_products"] = relevant_products
            result["response"] = self.generate_inquiry_response(email, relevant_products)

        return result

    def process_batch(self, emails_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Process every row of emails_df and return the four output DataFrames."""
        all_results = []
        for _, email_row in emails_df.iterrows():
            email = {
                "email_id": email_row["email_id"],
                "subject": email_row["subject"],
                "message": email_row["message"],
            }
            all_results.append(self.process_email(email))

        email_classification_data = []
        order_status_data = []
        order_response_data = []
        inquiry_response_data = []

        for result in all_results:
            email_id = result["email"]["email_id"]
            email_type = result["email_type"]
            response = result["response"]

            email_classification_data.append({"email ID": email_id, "category": email_type})

            if email_type == EmailType.ORDER_REQUEST.value:
                for item in result["order_items"]:
                    order_status_data.append({
                        "email ID": email_id,
                        "product ID": item.get("product_id"),
                        "quantity": item.get("quantity"),
                        "status": item.get("status"),
                    })
                order_response_data.append({"email ID": email_id, "response": response})
            else:
                inquiry_response_data.append({"email ID": email_id, "response": response})

        return {
            "email_classification_df": pd.DataFrame(email_classification_data),
            "order_status_df": pd.DataFrame(order_status_data, columns=["email ID", "product ID", "quantity", "status"]),
            "order_response_df": pd.DataFrame(order_response_data, columns=["email ID", "response"]),
            "inquiry_response_df": pd.DataFrame(inquiry_response_data, columns=["email ID", "response"]),
            "raw_results": all_results,
        }

    def process_customer_query(self, query: str, email_id: str = "CLI_001", use_agentic: bool = False) -> Dict[str, Any]:
        """
        Single-query entry point used by the chat interface.
        
        Args:
            query: The customer query
            email_id: Email identifier
            use_agentic: If True, use full agentic framework with planning and tool use
        """
        # If agentic mode is requested and available, use it
        if use_agentic and self.agent_orchestrator is not None:
            return self.process_agentic_query(query, email_id)
        
        # Otherwise, use standard processing
        email = {"email_id": email_id, "subject": "Chat Query", "message": query}
        result = self.process_email(email)

        formatted: Dict[str, Any] = {
            "Classification": result["email_type"],
            "Generated Response": result["response"],
            "Order Details (if applicable)": "N/A",
            "Relevant Products (if applicable)": "N/A",
        }

        if result["order_items"]:
            formatted["Order Details (if applicable)"] = "\n".join(
                f"- {item['quantity']}x {item['product_name']} (ID: {item['product_id']}, Status: {item['status']})"
                for item in result["order_items"]
            )
        if result["relevant_products"]:
            formatted["Relevant Products (if applicable)"] = "\n".join(
                f"- {p['name']} (ID: {p['product_id']}, Stock: {p['stock']}, Price: {p['price']} {p['currency']})"
                for p in result["relevant_products"]
            )

        return formatted
    
    def process_agentic_query(self, query: str, email_id: str = "CLI_001") -> Dict[str, Any]:
        """
        Process query using full agentic framework with autonomous planning and tool use.
        
        This method enables the AI to:
        - Plan the approach
        - Select and use tools dynamically
        - Reason through multiple steps
        - Self-correct and adapt
        
        Args:
            query: The customer query
            email_id: Email identifier
            
        Returns:
            Formatted result with agentic execution details
        """
        if self.agent_orchestrator is None:
            return {
                "error": "Agentic framework not available",
                "fallback": self.process_customer_query(query, email_id, use_agentic=False)
            }
        
        logger.info(f"Processing agentic query: {query}")
        
        # Context for the agent
        context = {
            "email_id": email_id,
            "available_products": len(self.products_df),
            "rag_enabled": self.rag_pipeline is not None
        }
        
        # Execute with multi-agent orchestration
        result = self.agent_orchestrator.execute_with_planning(query, context)
        
        # Format for display
        formatted = {
            "Mode": "🤖 Agentic (Autonomous AI)",
            "Query": query,
            "Generated Response": result["answer"],
            "Planning": {
                "Steps": result["plan"]["steps"],
                "Complexity": result["plan"]["complexity"],
                "Required Tools": result["plan"]["required_tools"]
            },
            "Execution": {
                "Iterations": result["execution"]["iterations"],
                "Tools Used": result["execution"]["tool_calls"],
                "Execution Trace": result["execution"]["trace"]
            },
            "Validation": {
                "Passed": result["validation"]["passed"],
                "Confidence": result["validation"]["confidence"]
            },
            "Success": result["success"]
        }
        
        # Apply guardrails to final answer
        violations = self.review_response_for_sensitive_leaks(result["answer"])
        if violations:
            logger.warning(f"Agentic response blocked by guardrail: {violations}")
            formatted["Guardrail Status"] = "BLOCKED"
            formatted["Violations"] = violations
            formatted["Generated Response"] = "Response blocked due to potential sensitive information leakage."
        else:
            formatted["Guardrail Status"] = "PASSED"
        
        return formatted


# --------------------------------------------------------------------------
# Export helpers
# --------------------------------------------------------------------------

def export_outputs(dfs: Dict[str, pd.DataFrame], output_dir: str = "generated_outputs") -> Dict[str, str]:
    """Write the four output DataFrames to a single xlsx workbook + individual CSVs.
    Returns a dict of the file paths written.
    """
    from pathlib import Path

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    workbook_path = out_dir / "business_problem_output.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        dfs["email_classification_df"].to_excel(writer, sheet_name="email-classification", index=False)
        dfs["order_status_df"].to_excel(writer, sheet_name="order-status", index=False)
        dfs["order_response_df"].to_excel(writer, sheet_name="order-response", index=False)
        dfs["inquiry_response_df"].to_excel(writer, sheet_name="inquiry-response", index=False)

    paths = {"workbook": str(workbook_path)}
    for key, filename in [
        ("email_classification_df", "email-classification.csv"),
        ("order_status_df", "order-status.csv"),
        ("order_response_df", "order-response.csv"),
        ("inquiry_response_df", "inquiry-response.csv"),
    ]:
        csv_path = out_dir / filename
        dfs[key].to_csv(csv_path, index=False)
        paths[key] = str(csv_path)

    return paths


def load_backend_from_env(products_df: pd.DataFrame) -> RetailAssistantBackend:
    """Convenience constructor that reads OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
    from the environment (loaded via python-dotenv in the app entrypoint)."""
    return RetailAssistantBackend(
        products_df=products_df,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    )
