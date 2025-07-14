Follow these steps to set up the project locally:

```bash
# 1. Clone the repository
git clone https://github.com/your-username/genai-summarizer.git
cd genai-summarizer

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
