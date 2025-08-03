"""
Celery configuration for academic paper scraping and analysis system.
"""

import os
from celery import Celery
from django.conf import settings

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'parsepaper.settings')

app = Celery('parsepaper')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django apps.
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)

# Celery configuration
app.conf.update(
    # Task routing
    task_routes={
        'papers.tasks.*': {'queue': 'papers'},
        'analysis.tasks.*': {'queue': 'analysis'},
        'scraping.tasks.*': {'queue': 'scraping'},
    },
    
    # Task serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task execution
    task_always_eager=False,
    task_eager_propagates=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    
    # Result backend
    result_backend='redis://localhost:6379/1',
    
    # Broker settings
    broker_url='redis://localhost:6379/0',
    broker_connection_retry_on_startup=True,
    
    # Task time limits
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,        # 10 minutes
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'scrape-recent-papers': {
            'task': 'papers.tasks.scrape_recent_papers',
            'schedule': 3600.0,  # Every hour
        },
        'analyze-pending-papers': {
            'task': 'analysis.tasks.analyze_pending_papers',
            'schedule': 1800.0,  # Every 30 minutes
        },
        'update-paper-metrics': {
            'task': 'papers.tasks.update_paper_metrics',
            'schedule': 7200.0,  # Every 2 hours
        },
    },
    
    # Task result settings
    task_ignore_result=False,
    task_store_errors_even_if_ignored=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

@app.task(bind=True)
def debug_task(self):
    """Debug task to test Celery configuration."""
    print(f'Request: {self.request!r}')
    return 'Celery is working!' 