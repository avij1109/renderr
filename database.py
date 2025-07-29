"""
MongoDB Database Models and Operations for PPG Health System
"""
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
import os
import time
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# Pydantic models for data validation
class SubjectCreate(BaseModel):
    subject_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    notes: Optional[str] = None

class MeasurementCreate(BaseModel):
    subject_id: str
    heart_rate: int
    heart_rate_confidence: int
    signal_quality: str
    bp_category: Optional[str] = None
    bp_confidence: Optional[int] = None
    measurement_duration: int  # in seconds
    frame_count: int

class DatabaseManager:
    def __init__(self):
        self.client = None
        self.database = None
        self.subjects_collection = None
        self.measurements_collection = None
        
    async def connect(self):
        """Connect to MongoDB Atlas"""
        try:
            # Get MongoDB connection string from environment variable
            mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
            
            self.client = AsyncIOMotorClient(mongodb_url)
            self.database = self.client.ppg_health_db
            self.subjects_collection = self.database.subjects
            self.measurements_collection = self.database.measurements
            
            # Create indexes for better performance
            await self.create_indexes()
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info("✅ Connected to MongoDB successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            return False
    
    async def create_indexes(self):
        """Create database indexes for better query performance"""
        try:
            # Index on subject_id for measurements
            await self.measurements_collection.create_index([("subject_id", ASCENDING)])
            
            # Index on timestamp for measurements (most recent first)
            await self.measurements_collection.create_index([("timestamp", DESCENDING)])
            
            # Compound index for subject + timestamp
            await self.measurements_collection.create_index([
                ("subject_id", ASCENDING),
                ("timestamp", DESCENDING)
            ])
            
            logger.info("📊 Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to create indexes: {e}")
    
    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            logger.info("🔌 Disconnected from MongoDB")
    
    # ============ SUBJECT OPERATIONS ============
    
    async def create_subject(self, subject_data: SubjectCreate) -> str:
        """Create a new subject and return the subject_id"""
        try:
            # Get the next subject ID number
            last_subject = await self.subjects_collection.find_one(
                {}, sort=[("subject_number", DESCENDING)]
            )
            
            next_number = 1 if not last_subject else last_subject["subject_number"] + 1
            subject_id = f"SUB{next_number:03d}"  # SUB001, SUB002, etc.
            
            subject_doc = {
                "subject_id": subject_id,
                "subject_number": next_number,
                "subject_name": subject_data.subject_name,
                "age": subject_data.age,
                "gender": subject_data.gender,
                "notes": subject_data.notes,
                "created_at": datetime.now(timezone.utc),
                "last_measurement": None,
                "total_measurements": 0
            }
            
            result = await self.subjects_collection.insert_one(subject_doc)
            logger.info(f"👤 Created new subject: {subject_id}")
            return subject_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create subject: {e}")
            raise
    
    async def get_all_subjects(self) -> List[Dict]:
        """Get all subjects ordered by subject_number"""
        try:
            cursor = self.subjects_collection.find(
                {}, 
                {"_id": 0}  # Exclude MongoDB _id field
            ).sort("subject_number", ASCENDING)
            
            subjects = await cursor.to_list(length=None)
            return subjects
            
        except Exception as e:
            logger.error(f"❌ Failed to get subjects: {e}")
            return []
    
    async def get_subject(self, subject_id: str) -> Optional[Dict]:
        """Get a specific subject by ID"""
        try:
            subject = await self.subjects_collection.find_one(
                {"subject_id": subject_id},
                {"_id": 0}
            )
            return subject
            
        except Exception as e:
            logger.error(f"❌ Failed to get subject {subject_id}: {e}")
            return None
    
    # ============ MEASUREMENT OPERATIONS ============
    
    async def save_measurement(self, measurement_data: MeasurementCreate) -> str:
        """Save a new measurement and return the measurement_id"""
        try:
            measurement_doc = {
                "measurement_id": f"M{int(time.time() * 1000)}",  # Unique ID with timestamp
                "subject_id": measurement_data.subject_id,
                "heart_rate": measurement_data.heart_rate,
                "heart_rate_confidence": measurement_data.heart_rate_confidence,
                "signal_quality": measurement_data.signal_quality,
                "bp_category": measurement_data.bp_category,
                "bp_confidence": measurement_data.bp_confidence,
                "measurement_duration": measurement_data.measurement_duration,
                "frame_count": measurement_data.frame_count,
                "timestamp": datetime.now(timezone.utc)
            }
            
            # Insert measurement
            result = await self.measurements_collection.insert_one(measurement_doc)
            
            # Update subject's last measurement and count
            await self.subjects_collection.update_one(
                {"subject_id": measurement_data.subject_id},
                {
                    "$set": {"last_measurement": datetime.now(timezone.utc)},
                    "$inc": {"total_measurements": 1}
                }
            )
            
            logger.info(f"💾 Saved measurement for subject {measurement_data.subject_id}")
            return measurement_doc["measurement_id"]
            
        except Exception as e:
            logger.error(f"❌ Failed to save measurement: {e}")
            raise
    
    async def get_subject_history(self, subject_id: str, limit: int = 50) -> List[Dict]:
        """Get measurement history for a subject"""
        try:
            cursor = self.measurements_collection.find(
                {"subject_id": subject_id},
                {"_id": 0}  # Exclude MongoDB _id field
            ).sort("timestamp", DESCENDING).limit(limit)
            
            measurements = await cursor.to_list(length=limit)
            return measurements
            
        except Exception as e:
            logger.error(f"❌ Failed to get history for {subject_id}: {e}")
            return []
    
    async def get_subject_stats(self, subject_id: str) -> Dict:
        """Get statistical summary for a subject"""
        try:
            pipeline = [
                {"$match": {"subject_id": subject_id}},
                {"$group": {
                    "_id": "$subject_id",
                    "total_measurements": {"$sum": 1},
                    "avg_heart_rate": {"$avg": "$heart_rate"},
                    "min_heart_rate": {"$min": "$heart_rate"},
                    "max_heart_rate": {"$max": "$heart_rate"},
                    "latest_measurement": {"$max": "$timestamp"},
                    "bp_categories": {"$push": "$bp_category"}
                }}
            ]
            
            result = await self.measurements_collection.aggregate(pipeline).to_list(1)
            
            if result:
                stats = result[0]
                # Count BP categories
                bp_counts = {}
                for bp in stats.get("bp_categories", []):
                    if bp:
                        bp_counts[bp] = bp_counts.get(bp, 0) + 1
                
                return {
                    "subject_id": subject_id,
                    "total_measurements": stats["total_measurements"],
                    "avg_heart_rate": round(stats["avg_heart_rate"], 1) if stats["avg_heart_rate"] else 0,
                    "min_heart_rate": stats["min_heart_rate"],
                    "max_heart_rate": stats["max_heart_rate"],
                    "latest_measurement": stats["latest_measurement"],
                    "bp_distribution": bp_counts
                }
            else:
                return {"subject_id": subject_id, "total_measurements": 0}
                
        except Exception as e:
            logger.error(f"❌ Failed to get stats for {subject_id}: {e}")
            return {"subject_id": subject_id, "total_measurements": 0}

# Global database manager instance
db_manager = DatabaseManager()
