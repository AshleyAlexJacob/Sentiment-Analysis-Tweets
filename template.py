import os

# Define folder structure
folders = [
    "notebooks",
    "src/data",
    "src/features",
    "src/models",
    "src/utils",
    "streamlit_app",
    "dagshub_wandb",
    "data/raw",
    "data/processed",
    "outputs/models",
    "outputs/reports"
]

def generate_project():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("✅ Project structure and boilerplate scripts created.")

if __name__ == "__main__":
    generate_project()