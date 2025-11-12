import os
from dotenv import load_dotenv

# 1️⃣ Load your AWS credentials
load_dotenv()

# 2️⃣ Ensure DVC can access them
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION")

# 3️⃣ Define your data directory or file
data_path = "data/"   # or "data/train.csv"

# 4️⃣ Run DVC + Git commands
os.system(f"dvc add {data_path}")
os.system("git add .")
os.system('git commit -m "Updated data and DVC tracking"')
os.system("dvc push")

print("✅ DVC data and code successfully updated & pushed!")
