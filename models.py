from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

def train_and_test(X, y):
    print("🧠 AI Models Training (Real EEG Data - Research Standard) Shuru Ho Rahi Hai...\n")
    
    # Scaling the real mathematical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Label Encoding (Dementia vs Healthy)
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    
    # Best Model for Medical Tabular Data
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
    
    # 10-Fold Cross Validation (Testing on all 176 patients systematically)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    predictions = []
    y_true = []
    
    print(f"Testing Strategy: {len(y)} Asli Patients par 10-Fold Cross-Validation chal rahi hai...")
    
    for train_idx, test_idx in cv.split(X_scaled, y_int):
        X_train_fold, X_test_fold = X_scaled[train_idx], X_scaled[test_idx]
        y_train_fold, y_test_fold = y_int[train_idx], y_int[test_idx]
        
        # SMOTE sirf training data par taake classes balance rahein
        smote = SMOTE(random_state=42)
        try:
            X_train_smote, y_train_smote = smote.fit_resample(X_train_fold, y_train_fold)
        except:
            X_train_smote, y_train_smote = X_train_fold, y_train_fold
            
        rf_model.fit(X_train_smote, y_train_smote)
        fold_preds = rf_model.predict(X_test_fold)
        
        predictions.extend(fold_preds)
        y_true.extend(y_test_fold)
    
    # Final Presentation Results
    print("==================================================")
    print("🌟 Final Results (Real Alzheimer's Data)")
    print("==================================================")
    print(classification_report(y_true, predictions, target_names=le.classes_.astype(str)))
    
    acc = accuracy_score(y_true, predictions) * 100
    print(f"\n🏆 Asli Data Accuracy: {acc:.2f}%\n")