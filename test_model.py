import joblib
import pandas as pd

# Test model loading
try:
    pipe = joblib.load('pipe.pkl')
    print("✅ Model loaded successfully!")
    print(f"Model type: {type(pipe)}")
    
    if hasattr(pipe, 'named_steps'):
        print(f"Pipeline steps: {list(pipe.named_steps.keys())}")
    
    # Test with sample data
    test_data = pd.DataFrame({
        'batting_team': ['Mumbai Indians'],
        'bowling_team': ['Chennai Super Kings'],
        'city': ['Mumbai'],
        'runs_left': [50],
        'balls_left': [60],
        'wickets': [5],
        'total_runs_x': [120],
        'crr': [8.0],
        'rrr': [5.0]
    })
    
    print(f"Test data shape: {test_data.shape}")
    print(f"Test data columns: {list(test_data.columns)}")
    
    if hasattr(pipe, 'predict_proba'):
        result = pipe.predict_proba(test_data)
        print(f"✅ Prediction successful: {result}")
    else:
        pred = pipe.predict(test_data)
        print(f"✅ Prediction successful: {pred}")
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc() 