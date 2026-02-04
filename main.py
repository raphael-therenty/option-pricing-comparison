import os
import sys
import subprocess
import argparse

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def run_streamlit():
    app_path = os.path.join(os.path.dirname(__file__), 'app', 'streamlit_app.py')
    subprocess.run(['streamlit', 'run', app_path])

def main() :
    parser = argparse.ArgumentParser(description= 'Option Pricing Project')
    parser.add_argument('--app', action= 'store_true', help= 'Run the Streamlit app')
    args = parser.parse_args()

    if args.app :
        run_streamlit()
    else :
        print('Usage: python main.py --app')

if __name__ == '__main__' :
    main()
    
