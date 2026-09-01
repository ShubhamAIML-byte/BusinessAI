"""
database.py
-----------
SQLAlchemy ORM models and database operations for persisting query history.

This module provides:
- SQLite database with SQLAlchemy ORM
- QueryHistory model for storing all query interactions
- CRUD operations for query history
- Automatic database initialization
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
import json

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Database setup
DATABASE_URL = "sqlite:///./business_ai.db"
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class QueryHistory(Base):
    """Store all query interactions with AI system."""
    __tablename__ = "query_history"
    
    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    type = Column(String(50), nullable=False)  # EmailType: "order request" or "product inquiry"
    confidence = Column(Integer, nullable=False)
    status = Column(String(20), nullable=False)  # "Passed" or "Blocked"
    timestamp = Column(DateTime, default=datetime.now, nullable=False)
    duration = Column(Float, nullable=False)
    agents = Column(Text, nullable=False)  # JSON array of agent names
    response = Column(Text, nullable=False)
    guardrail_score = Column(Integer, nullable=False)
    guardrail_checks = Column(Text, nullable=False)  # JSON object
    order_details = Column(Text, nullable=True)
    relevant_products = Column(Text, nullable=True)
    violations = Column(Text, nullable=True)  # JSON array


# -----------------------------------------------------------------------------
# Database initialization
# -----------------------------------------------------------------------------

def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # Don't close here, let caller handle it


# -----------------------------------------------------------------------------
# CRUD Operations
# -----------------------------------------------------------------------------

def create_query_record(
    query: str,
    type: str,
    confidence: int,
    status: str,
    timestamp: datetime,
    duration: float,
    agents: List[str],
    response: str,
    guardrail_score: int,
    guardrail_checks: Dict[str, bool],
    order_details: Optional[str] = None,
    relevant_products: Optional[str] = None,
    violations: Optional[List[str]] = None
) -> QueryHistory:
    """Create a new query history record."""
    db = SessionLocal()
    try:
        record = QueryHistory(
            query=query,
            type=type,
            confidence=confidence,
            status=status,
            timestamp=timestamp,
            duration=duration,
            agents=json.dumps(agents),
            response=response,
            guardrail_score=guardrail_score,
            guardrail_checks=json.dumps(guardrail_checks),
            order_details=order_details,
            relevant_products=relevant_products,
            violations=json.dumps(violations or [])
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    finally:
        db.close()


def get_all_queries(limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:
    """Get all query history records, newest first."""
    db = SessionLocal()
    try:
        query = db.query(QueryHistory).order_by(QueryHistory.timestamp.desc())
        
        if limit:
            query = query.limit(limit).offset(offset)
        
        records = query.all()
        
        # Convert to dict format matching app.py session_state structure
        return [
            {
                "id": record.id,
                "query": record.query,
                "type": record.type,
                "confidence": record.confidence,
                "status": record.status,
                "timestamp": record.timestamp,
                "duration": record.duration,
                "agents": json.loads(record.agents),
                "response": record.response,
                "guardrail_score": record.guardrail_score,
                "guardrail_checks": json.loads(record.guardrail_checks),
                "order_details": record.order_details,
                "relevant_products": record.relevant_products,
                "violations": json.loads(record.violations) if record.violations else []
            }
            for record in records
        ]
    finally:
        db.close()


def get_query_by_id(query_id: int) -> Optional[Dict[str, Any]]:
    """Get a single query record by ID."""
    db = SessionLocal()
    try:
        record = db.query(QueryHistory).filter(QueryHistory.id == query_id).first()
        if not record:
            return None
        
        return {
            "id": record.id,
            "query": record.query,
            "type": record.type,
            "confidence": record.confidence,
            "status": record.status,
            "timestamp": record.timestamp,
            "duration": record.duration,
            "agents": json.loads(record.agents),
            "response": record.response,
            "guardrail_score": record.guardrail_score,
            "guardrail_checks": json.loads(record.guardrail_checks),
            "order_details": record.order_details,
            "relevant_products": record.relevant_products,
            "violations": json.loads(record.violations) if record.violations else []
        }
    finally:
        db.close()


def get_queries_by_type(query_type: str) -> List[Dict[str, Any]]:
    """Get all queries of a specific type."""
    db = SessionLocal()
    try:
        records = db.query(QueryHistory)\
            .filter(QueryHistory.type == query_type)\
            .order_by(QueryHistory.timestamp.desc())\
            .all()
        
        return [
            {
                "id": record.id,
                "query": record.query,
                "type": record.type,
                "confidence": record.confidence,
                "status": record.status,
                "timestamp": record.timestamp,
                "duration": record.duration,
                "agents": json.loads(record.agents),
                "response": record.response,
                "guardrail_score": record.guardrail_score,
                "guardrail_checks": json.loads(record.guardrail_checks),
                "order_details": record.order_details,
                "relevant_products": record.relevant_products,
                "violations": json.loads(record.violations) if record.violations else []
            }
            for record in records
        ]
    finally:
        db.close()


def get_queries_by_status(status: str) -> List[Dict[str, Any]]:
    """Get all queries with a specific status."""
    db = SessionLocal()
    try:
        records = db.query(QueryHistory)\
            .filter(QueryHistory.status == status)\
            .order_by(QueryHistory.timestamp.desc())\
            .all()
        
        return [
            {
                "id": record.id,
                "query": record.query,
                "type": record.type,
                "confidence": record.confidence,
                "status": record.status,
                "timestamp": record.timestamp,
                "duration": record.duration,
                "agents": json.loads(record.agents),
                "response": record.response,
                "guardrail_score": record.guardrail_score,
                "guardrail_checks": json.loads(record.guardrail_checks),
                "order_details": record.order_details,
                "relevant_products": record.relevant_products,
                "violations": json.loads(record.violations) if record.violations else []
            }
            for record in records
        ]
    finally:
        db.close()


def get_query_stats() -> Dict[str, Any]:
    """Get aggregate statistics about queries."""
    db = SessionLocal()
    try:
        total = db.query(QueryHistory).count()
        
        if total == 0:
            return {
                "total": 0,
                "passed": 0,
                "blocked": 0,
                "avg_confidence": 0,
                "avg_guardrail_score": 0,
                "by_type": {}
            }
        
        passed = db.query(QueryHistory).filter(QueryHistory.status == "Passed").count()
        blocked = db.query(QueryHistory).filter(QueryHistory.status == "Blocked").count()
        
        # Average confidence
        from sqlalchemy import func
        avg_confidence = db.query(func.avg(QueryHistory.confidence)).scalar() or 0
        avg_guardrail = db.query(func.avg(QueryHistory.guardrail_score)).scalar() or 0
        
        # Count by type
        order_count = db.query(QueryHistory).filter(QueryHistory.type == "order request").count()
        inquiry_count = db.query(QueryHistory).filter(QueryHistory.type == "product inquiry").count()
        
        return {
            "total": total,
            "passed": passed,
            "blocked": blocked,
            "avg_confidence": round(avg_confidence, 1),
            "avg_guardrail_score": round(avg_guardrail, 1),
            "by_type": {
                "order request": order_count,
                "product inquiry": inquiry_count
            }
        }
    finally:
        db.close()


def delete_query(query_id: int) -> bool:
    """Delete a query record by ID."""
    db = SessionLocal()
    try:
        record = db.query(QueryHistory).filter(QueryHistory.id == query_id).first()
        if record:
            db.delete(record)
            db.commit()
            return True
        return False
    finally:
        db.close()


def clear_all_queries() -> int:
    """Delete all query records. Returns count of deleted records."""
    db = SessionLocal()
    try:
        count = db.query(QueryHistory).count()
        db.query(QueryHistory).delete()
        db.commit()
        return count
    finally:
        db.close()


# -----------------------------------------------------------------------------
# Initialize database on import
# -----------------------------------------------------------------------------

init_db()
