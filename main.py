from data_loader import get_prepared_data
from models import train_and_test

def main():
    print("Step 1: Data load kiya ja raha hai...")
    X, y = get_prepared_data()
    
    if X is not None and y is not None:
        print(f"Data Successfully Loaded! Total Patients Ready for AI: {len(X)}")
        print("\nStep 2: AI models ki taraf rawana...")
        train_and_test(X, y)
    else:
        print("Error: Data load nahi ho saka. Path verify karein.")

if __name__ == "__main__":
    main()