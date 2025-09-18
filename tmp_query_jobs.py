from app.firebase_queue import get_firestore_client
from google.cloud import firestore

client = get_firestore_client()
q = client.collection('jobs').where('status', '==', 'queued').order_by('createdAt').limit(3)
print('Query ready')
print(list(q.stream()))
