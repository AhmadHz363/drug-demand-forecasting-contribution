import sys
sys.path.insert(0, 'src')

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.services.overview_service import get_overview_stats
from app.core.config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

try:
    stats = get_overview_stats(db)
    print("✓ Overview service works!")
    print(f"  Total drugs: {stats['total_drugs']}")
    print(f"  Total categories: {stats['total_categories']}")
    print(f"  Total receipts: {stats['total_receipts']}")
    print(f"  Recent drugs: {len(stats['recent_drugs'])}")
    print(f"  Recent categories: {len(stats['recent_categories'])}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    db.close()
