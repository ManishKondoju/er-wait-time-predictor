import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_er_data(n_records=5000):
    """Generate synthetic ER visit data"""
    np.random.seed(42)
    random.seed(42)
    
    # Define complaints with wait time ranges
    complaints = {
        'Chest Pain': (30, 90),
        'Abdominal Pain': (60, 180),
        'Breathing Difficulty': (20, 60),
        'Fracture/Injury': (45, 150),
        'Fever': (90, 240),
        'Headache': (120, 300),
        'Minor Cut/Wound': (90, 240),
        'Other': (60, 180)
    }
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(n_records):
        # Generate random timestamp in 2023
        random_days = random.randint(0, 364)
        random_seconds = random.randint(0, 86399)
        timestamp = start_date + timedelta(days=random_days, seconds=random_seconds)
        
        # Select random complaint
        complaint = random.choice(list(complaints.keys()))
        min_wait, max_wait = complaints[complaint]
        
        # Base wait time
        hour = timestamp.hour
        base_wait = random.randint(min_wait, max_wait)
        
        # Apply time-based multipliers
        wait_multiplier = 1.0
        if 18 <= hour <= 23:  # Evening rush
            wait_multiplier = 1.3
        elif 12 <= hour <= 14:  # Lunch time
            wait_multiplier = 1.2
        
        # Weekend multiplier
        if timestamp.weekday() in [5, 6]:
            wait_multiplier *= 1.25
            
        wait_time = int(base_wait * wait_multiplier)
        
        # Generate other features
        age = min(95, max(1, int(np.random.normal(45, 20))))
        triage_level = random.choices([1, 2, 3, 4, 5], weights=[5, 15, 40, 30, 10])[0]
        arrival_mode = random.choices(['Walk-in', 'Ambulance'], weights=[85, 15])[0]
        occupancy = random.randint(40, 100)
        
        # Ambulance reduces wait time
        if arrival_mode == 'Ambulance':
            wait_time = int(wait_time * 0.5)
        
        # High occupancy increases wait time
        if occupancy > 85:
            wait_time = int(wait_time * 1.2)
        
        data.append({
            'timestamp': timestamp,
            'hour': hour,
            'day_of_week': timestamp.weekday(),
            'month': timestamp.month,
            'complaint': complaint,
            'age': age,
            'triage_level': triage_level,
            'arrival_mode': arrival_mode,
            'current_occupancy': occupancy,
            'wait_time_minutes': wait_time
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Test the generator
    df = generate_er_data(100)
    print(df.head())
    print(f"\nGenerated {len(df)} records")
    print(f"Average wait time: {df['wait_time_minutes'].mean():.1f} minutes")