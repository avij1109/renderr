"""
Test MongoDB Atlas connection
"""
import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_connection():
    # Get MongoDB connection string from environment variable
    MONGODB_URL = os.getenv("MONGODB_URL")
    
    if not MONGODB_URL:
        print("❌ No MongoDB connection string found in environment variables")
        print("Please create a .env file with MONGODB_URL")
        return False
    
    try:
        # Connect to MongoDB
        client = AsyncIOMotorClient(MONGODB_URL)
        
        # Test connection
        await client.admin.command('ping')
        logger.info("✅ Successfully connected to MongoDB Atlas!")
        
        # Test database creation
        database = client.ppg_health_db
        subjects_collection = database.subjects
        
        # Test insert (this will create the database and collection)
        test_doc = {
            "test": True,
            "message": "Connection test successful"
        }
        result = await subjects_collection.insert_one(test_doc)
        logger.info(f"✅ Test document inserted with ID: {result.inserted_id}")
        
        # Clean up test document
        await subjects_collection.delete_one({"_id": result.inserted_id})
        logger.info("✅ Test document cleaned up")
        
        # Close connection
        client.close()
        logger.info("✅ MongoDB Atlas connection test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB Atlas: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_connection())
