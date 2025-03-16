from model import train_model
from analysis import analyze_team_performance

if __name__ == "__main__":
    print("Choose an option:")
    print("1 - Analyze Team Performance")
    print("2 - Predict Win Percentage")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        analyze_team_performance()
    elif choice == "2":
        print("\nRunning ML Model...")
        train_model()
    else:
        print("Invalid choice. Please enter 1 or 2.")
