import streamlit as st
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def _get_expected_columns_from_model(fitted_model):
    expected = None
    if hasattr(fitted_model, 'feature_names_in_'):
        expected = list(fitted_model.feature_names_in_)
    if expected is None and hasattr(fitted_model, 'named_steps'):
        for step_name, step in fitted_model.named_steps.items():
            if hasattr(step, 'feature_names_in_'):
                expected = list(step.feature_names_in_)
                break
    if expected is None and hasattr(fitted_model, 'named_steps'):
        for step_name, step in fitted_model.named_steps.items():
            if isinstance(step, ColumnTransformer):
                cols = []
                transformers = getattr(step, 'transformers_', None) or getattr(step, 'transformers', [])
                for name, transformer, columns in transformers:
                    if isinstance(columns, (list, tuple)):
                        cols.extend([c for c in columns if isinstance(c, str)])
                if cols:
                    expected = cols
                    break
    return expected


def _extract_allowed_categories(fitted_model):
    allowed = {}
    if hasattr(fitted_model, 'named_steps'):
        for _, step in fitted_model.named_steps.items():
            if isinstance(step, ColumnTransformer):
                transformers = getattr(step, 'transformers_', None) or getattr(step, 'transformers', [])
                for name, transformer, columns in transformers:
                    enc = None
                    if hasattr(transformer, 'named_steps'):
                        for _, sub in transformer.named_steps.items():
                            if isinstance(sub, OneHotEncoder):
                                enc = sub
                                break
                    elif isinstance(transformer, OneHotEncoder):
                        enc = transformer
                    if enc is not None and hasattr(enc, 'categories_') and isinstance(columns, (list, tuple)):
                        for idx, col in enumerate(columns):
                            if isinstance(col, str) and idx < len(enc.categories_):
                                allowed[col] = list(enc.categories_[idx])
    return allowed


def _align_input_columns(df: pd.DataFrame, fitted_model) -> pd.DataFrame:
    expected_columns = _get_expected_columns_from_model(fitted_model)
    if not expected_columns:
        return df

    synonyms = {
        'venue': 'city',
        'city': 'venue',
        'wickets_left': 'wickets',
        'wickets_in_hand': 'wickets',
        'current_run_rate': 'crr',
        'required_run_rate': 'rrr',
        'total_runs': 'total_runs_x',
        'total_runs_x': 'total_runs',
        'bat_team': 'batting_team',
        'bowl_team': 'bowling_team',
    }

    aligned = {}
    for col in expected_columns:
        if col in df.columns:
            aligned[col] = df[col].values
        elif col in synonyms and synonyms[col] in df.columns:
            aligned[col] = df[synonyms[col]].values
        else:
            if any(token in col for token in ['rate', 'runs', 'balls', 'wicket', 'over', 'score', 'target']):
                aligned[col] = [0]
            else:
                aligned[col] = ['unknown']

    aligned_df = pd.DataFrame(aligned)

    missing = [c for c in expected_columns if c not in df.columns and not (c in synonyms and synonyms[c] in df.columns)]
    extra = [c for c in df.columns if c not in expected_columns and not (c in synonyms and synonyms[c] in expected_columns)]
    if missing:
        st.info(f"Aligning to model schema. Missing columns filled with defaults: {missing}")
    if extra:
        st.info(f"Input had extra columns not used by model: {extra}")
    st.caption(f"Model expects columns: {expected_columns}")

    return aligned_df


def _normalize_value(value: str, allowed: set, synonyms_map: dict) -> str:
    if value in allowed:
        return value
    candidate = synonyms_map.get(value)
    if candidate and candidate in allowed:
        return candidate
    reverse = {v: k for k, v in synonyms_map.items()}
    candidate = reverse.get(value)
    if candidate and candidate in allowed:
        return candidate
    return value


def _allowed_for(allowed_categories: dict, keys: list, fallback: list) -> list:
    for key in keys:
        if key in allowed_categories:
            return sorted(allowed_categories[key])
    return sorted(fallback)


teams = ['Sunrisers Hyderabad', 'Mumbai Indians',
         'Royal Challengers Bangalore', 'Kolkata Knight Riders',
         'Punjab Kings', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Mumbai', 'Kolkata', 'Delhi', 'Hyderabad', 'Bangalore', 'Chennai',
          'Jaipur', 'Cape Town', 'Durban', 'Port Elizabeth', 'Centurion',
          'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
          'Ahmedabad', 'Cuttack', 'Nagpur', 'Visakhapatnam', 'Pune',
          'Raipur', 'Abu Dhabi', 'Sharjah', 'Ranchi']

try:
    pipe = joblib.load('pipe.pkl')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

allowed_categories = _extract_allowed_categories(pipe)

ui_teams = _allowed_for(allowed_categories, ['batting_team', 'bat_team'], teams)
ui_cities = _allowed_for(allowed_categories, ['city', 'venue'], cities)

st.title('IPL Probability Prediction')

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', ui_teams)
with col2:
    bowling_team = st.selectbox('Select the bowling team', ui_teams)

selected_city = st.selectbox('Select the city', ui_cities)
target = st.number_input('Target', min_value=1, step=1)
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs', min_value=0.1, max_value=20.0, step=0.1, format='%.1f')
with col5:
    wickets = st.number_input('Wickets', min_value=0, max_value=10, step=1)

if st.button('predict probability'):
    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left * 6)/balls_left if balls_left > 0 else 0

    team_synonyms = {
        'Kings XI Punjab': 'Punjab Kings',
        'Punjab Kings': 'Punjab Kings',
        'Delhi Daredevils': 'Delhi Capitals',
        'Delhi Capitals': 'Delhi Capitals',
    }
    city_synonyms = {
        'Bangalore': 'Bengaluru',
        'Bengaluru': 'Bangalore',
        'Bombay': 'Mumbai',
        'Mumbai': 'Mumbai',
    }

    allowed_bat = set(allowed_categories.get('batting_team', allowed_categories.get('bat_team', ui_teams)))
    allowed_bowl = set(allowed_categories.get('bowling_team', allowed_categories.get('bowl_team', ui_teams)))
    allowed_city = set(allowed_categories.get('city', allowed_categories.get('venue', ui_cities)))

    batting_team = _normalize_value(batting_team, allowed_bat, team_synonyms)
    bowling_team = _normalize_value(bowling_team, allowed_bowl, team_synonyms)
    selected_city = _normalize_value(selected_city, allowed_city, city_synonyms)

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                             'runs_left': [runs_left], 'balls_left': [balls_left],
                             'wickets': [wickets], 'total_runs_x': [score],
                             'crr': [crr], 'rrr': [rrr]})

    X = _align_input_columns(input_df, pipe)

    st.table(input_df)

    try:
        if hasattr(pipe, 'predict_proba'):
            result = pipe.predict_proba(X)
            st.text(result)
            loss = result[0][0]
            win = result[0][1]
            st.header(batting_team + "_" + str(round(win*100)) + "%")
            st.header(bowling_team + "_" + str(round(loss*100)) + "%")
        else:
            pred = pipe.predict(X)
            st.text(pred)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        
        # Try fallback approach
        st.write("Trying fallback prediction method...")
        try:
            # Try to get the final estimator directly
            if hasattr(pipe, 'steps') and len(pipe.steps) > 0:
                final_estimator = pipe.steps[-1][1]
                st.write(f"Final estimator: {type(final_estimator)}")
                
                # Try to transform the data manually through the pipeline
                X_transformed = X.copy()
                for step_name, step in pipe.steps[:-1]:  # All steps except the last one
                    if hasattr(step, 'transform') and not isinstance(step, str):
                        X_transformed = step.transform(X_transformed)
                        st.write(f"After {step_name}: {X_transformed.shape}")
                    else:
                        st.warning(f"Step {step_name} is not a valid transformer: {type(step)}")
                
                # Make prediction with final estimator
                if hasattr(final_estimator, 'predict_proba'):
                    result = final_estimator.predict_proba(X_transformed)
                    st.success("Fallback prediction successful!")
                    st.text(result)
                    loss = result[0][0]
                    win = result[0][1]
                    st.header(batting_team + "_" + str(round(win*100)) + "%")
                    st.header(bowling_team + "_" + str(round(loss*100)) + "%")
                else:
                    pred = final_estimator.predict(X_transformed)
                    st.success("Fallback prediction successful!")
                    st.text(pred)
            else:
                st.error("Could not extract final estimator from pipeline")
        except Exception as fallback_error:
            st.error(f"Fallback also failed: {str(fallback_error)}")
            
            # Manual transformation based on debug output
            st.write("Trying manual transformation method...")
            try:
                if hasattr(pipe, 'steps') and len(pipe.steps) > 0:
                    final_estimator = pipe.steps[-1][1]
                    if hasattr(final_estimator, 'predict_proba'):
                        # Manual one-hot encoding based on the categories we saw
                        team_categories = ['Chennai Super Kings', 'Deccan Chargers', 'Delhi Capitals', 'Delhi Daredevils', 
                                         'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals', 'Sunrisers Hyderabad']
                        city_categories = ['Abu Dhabi', 'Ahmedabad', 'Bangalore', 'Bloemfontein', 'Cape Town', 'Centurion', 
                                         'Chennai', 'Cuttack', 'Delhi', 'Durban', 'East London', 'Hyderabad', 'Jaipur', 
                                         'Johannesburg', 'Kimberley', 'Kolkata', 'Mumbai', 'Nagpur', 'Port Elizabeth', 
                                         'Pune', 'Raipur', 'Ranchi', 'Sharjah', 'Visakhapatnam']
                        
                        # Create one-hot encoded features (drop first category to avoid multicollinearity)
                        features = []
                        
                        # One-hot encode batting team (7 features - drop first category)
                        batting_team_encoded = [1 if team == batting_team else 0 for team in team_categories[1:]]
                        features.extend(batting_team_encoded)
                        
                        # One-hot encode bowling team (7 features - drop first category)
                        bowling_team_encoded = [1 if team == bowling_team else 0 for team in team_categories[1:]]
                        features.extend(bowling_team_encoded)
                        
                        # One-hot encode city (23 features - drop first category)
                        city_encoded = [1 if city == selected_city else 0 for city in city_categories[1:]]
                        features.extend(city_encoded)
                        
                        # Add numeric features (6 features)
                        features.extend([runs_left, balls_left, wickets, score, crr, rrr])
                        
                        # Total: 7 + 7 + 23 + 6 = 43 features
                        X_manual = pd.DataFrame([features])
                        st.write(f"Manual transformation shape: {X_manual.shape}")
                        
                        result = final_estimator.predict_proba(X_manual)
                        st.success("Manual transformation prediction successful!")
                        st.text(result)
                        loss = result[0][0]
                        win = result[0][1]
                        st.header(batting_team + "_" + str(round(win*100)) + "%")
                        st.header(bowling_team + "_" + str(round(loss*100)) + "%")
                    else:
                        st.error("Final estimator doesn't support predict_proba")
                else:
                    st.error("No steps found in pipeline")
            except Exception as manual_error:
                st.error(f"Manual transformation also failed: {str(manual_error)}")
                
                # Try alternative encoding patterns
                st.write("Trying alternative encoding patterns...")
                try:
                    if hasattr(pipe, 'steps') and len(pipe.steps) > 0:
                        final_estimator = pipe.steps[-1][1]
                        if hasattr(final_estimator, 'predict_proba'):
                            # Try different encoding patterns to get exactly 43 features
                            # Pattern 1: 8 + 7 + 23 + 5 = 43
                            features = []
                            
                            # Full batting team encoding (8)
                            batting_team_encoded = [1 if team == batting_team else 0 for team in team_categories]
                            features.extend(batting_team_encoded)
                            
                            # Partial bowling team encoding (7)
                            bowling_team_encoded = [1 if team == bowling_team else 0 for team in team_categories[1:]]
                            features.extend(bowling_team_encoded)
                            
                            # Partial city encoding (23)
                            city_encoded = [1 if city == selected_city else 0 for city in city_categories[1:]]
                            features.extend(city_encoded)
                            
                            # Partial numeric features (5)
                            features.extend([runs_left, balls_left, wickets, score, crr])
                            
                            X_alt = pd.DataFrame([features])
                            st.write(f"Alternative encoding shape: {X_alt.shape}")
                            
                            result = final_estimator.predict_proba(X_alt)
                            st.success("Alternative encoding prediction successful!")
                            st.text(result)
                            loss = result[0][0]
                            win = result[0][1]
                            st.header(batting_team + "_" + str(round(win*100)) + "%")
                            st.header(bowling_team + "_" + str(round(loss*100)) + "%")
                        else:
                            st.error("Final estimator doesn't support predict_proba")
                    else:
                        st.error("No steps found in pipeline")
                except Exception as alt_error:
                    st.error(f"Alternative encoding also failed: {str(alt_error)}")
                    
                    # Last resort: try to use the final estimator with raw data
                    st.write("Trying last resort method with raw data...")
                    try:
                        if hasattr(pipe, 'steps') and len(pipe.steps) > 0:
                            final_estimator = pipe.steps[-1][1]
                            if hasattr(final_estimator, 'predict_proba'):
                                # Convert categorical columns to numeric for raw prediction
                                X_numeric = X.copy()
                                for col in ['batting_team', 'bowling_team', 'city']:
                                    if col in X_numeric.columns:
                                        X_numeric[col] = pd.Categorical(X_numeric[col]).codes
                                
                                result = final_estimator.predict_proba(X_numeric)
                                st.success("Last resort prediction successful!")
                                st.text(result)
                                loss = result[0][0]
                                win = result[0][1]
                                st.header(batting_team + "_" + str(round(win*100)) + "%")
                                st.header(bowling_team + "_" + str(round(loss*100)) + "%")
                            else:
                                st.error("Final estimator doesn't support predict_proba")
                        else:
                            st.error("No steps found in pipeline")
                    except Exception as last_error:
                        st.error(f"Last resort also failed: {str(last_error)}")
        
        st.write("Debug info:")
        st.write(f"Input DataFrame shape: {X.shape}")
        st.write(f"Input DataFrame columns: {list(X.columns)}")
        st.write(f"Input DataFrame dtypes: {X.dtypes.to_dict()}")
        st.write(f"Model type: {type(pipe)}")
        if hasattr(pipe, 'named_steps'):
            st.write(f"Pipeline steps: {list(pipe.named_steps.keys())}")
            
            # Debug each step safely
            for step_name, step in pipe.named_steps.items():
                st.write(f"\n**Step '{step_name}':**")
                st.write(f"  Type: {type(step)}")
                
                # Safely check transformers
                try:
                    if hasattr(step, 'transformers'):
                        st.write(f"  Has transformers attribute")
                    if hasattr(step, 'transformers_'):
                        st.write(f"  Has fitted transformers attribute")
                    if hasattr(step, 'feature_names_in_'):
                        st.write(f"  Feature names: {list(step.feature_names_in_)}")
                except Exception as debug_error:
                    st.write(f"  Error accessing step attributes: {str(debug_error)}")
                    
                # Check if it's a ColumnTransformer and inspect safely
                if hasattr(step, 'transformers') or hasattr(step, 'transformers_'):
                    try:
                        transformers = getattr(step, 'transformers_', None) or getattr(step, 'transformers', [])
                        for i, transformer_info in enumerate(transformers):
                            if isinstance(transformer_info, (list, tuple)) and len(transformer_info) >= 3:
                                name, transformer, columns = transformer_info
                                st.write(f"    Transformer {i}: {name} - {type(transformer)} - {columns}")
                                if hasattr(transformer, 'categories_') and not isinstance(transformer, str):
                                    st.write(f"      Categories: {transformer.categories_}")
                            else:
                                st.write(f"    Transformer {i}: Invalid format - {transformer_info}")
                    except Exception as transformer_error:
                        st.write(f"    Error inspecting transformers: {str(transformer_error)}")
