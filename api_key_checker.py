import os
import sys
import time
import re
from typing import List, Tuple

# --- Dependency Check ---
try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
    from dotenv import load_dotenv
except ImportError:
    print("Error: Required libraries are not installed.")
    print("Please run the following command to install dependencies for this script:")
    print("pip install python-dotenv google-generativeai")
    sys.exit(1)

# ANSI color codes for colorful output in the terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Gemini API Key format regex, matches keys like "AIzaSy..."
GEMINI_KEY_PATTERN = re.compile(r"^AIzaSy[A-Za-z0-9_-]{33}$")

def check_gemini_api_key(api_key: str) -> Tuple[str, str]:
    """
    Checks if a single Google Gemini API key can return a value (has quota)
    for the specified model.
    """
    try:
        genai.configure(api_key=api_key)
        # [修正] 使用更通用的模型进行测试
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        model.generate_content(
            'test',
            generation_config=genai.types.GenerationConfig(max_output_tokens=1)
        )
        return 'valid', 'Key is valid and has sufficient quota for gemini-1.5-pro-latest.'
    except (google_exceptions.PermissionDenied, google_exceptions.Unauthenticated) as e:
        return 'permanently invalid', f'Authentication failed. Key is invalid or disabled. (Reason: {getattr(e, "message", str(e))})'
    except google_exceptions.ResourceExhausted as e:
        return 'temporarily invalid', f'Quota exceeded or rate limit hit. (Reason: {getattr(e, "message", str(e))})'
    except google_exceptions.DeadlineExceeded:
        return 'temporarily invalid', 'Request timed out.'
    except google_exceptions.ServiceUnavailable:
        return 'temporarily invalid', 'The Google AI service is currently unavailable.'
    except google_exceptions.GoogleAPICallError as e:
        if 'model not found' in str(e).lower() or 'permission' in str(e).lower():
             return 'permanently invalid', f'Model gemini-1.5-pro-latest not found or not accessible. (Reason: {getattr(e, "message", str(e))})'
        return 'permanently invalid', f'An unexpected API error occurred (Code: {getattr(e, "code", "N/A")}). (Reason: {getattr(e, "message", str(e))})'
    except Exception as e:
        return 'temporarily invalid', f'An unexpected network or client-side error occurred: {e}'

def update_env_file(keys_to_keep: List[str], env_file_path: str):
    # ... 此函数无需修改 ...
    original_env_path = env_file_path
    new_env_path = f"{original_env_path}.updated"
    updated_keys_str = ",".join(keys_to_keep)
    try:
        if not os.path.exists(original_env_path):
            print(f"{bcolors.FAIL}Error: Original .env file not found at {original_env_path}{bcolors.ENDC}")
            return
        target_dir = os.path.dirname(new_env_path) if os.path.dirname(new_env_path) else '.'
        if not os.access(target_dir, os.W_OK):
            print(f"{bcolors.FAIL}Error: No write permission for directory {target_dir}{bcolors.ENDC}")
            return
        with open(original_env_path, 'r') as f_in, open(new_env_path, 'w') as f_out:
            for line in f_in:
                if line.strip().startswith('GEMINI_API_KEYS='):
                    f_out.write(f'GEMINI_API_KEYS={updated_keys_str}\n')
                else:
                    f_out.write(line)
        print(f"\n{bcolors.OKGREEN}✅ Success! A new file `{new_env_path}` has been created.{bcolors.ENDC}")
        print(f"Please review it and rename to `.env` if correct.")
    except Exception as e:
        print(f"\n{bcolors.FAIL}Error writing updated .env file: {e}{bcolors.ENDC}")


def find_env_file():
    # ... 此函数无需修改 ...
    possible_paths = [
        '/app/.env',
        './.env',
        '.env',
        '/data/.env'
    ]
    for path in possible_paths:
        if os.path.exists(path):
            abs_path = os.path.abspath(path)
            print(f"{bcolors.OKCYAN}Found .env file at: {abs_path}{bcolors.ENDC}")
            return abs_path
    print(f"{bcolors.FAIL}Error: No .env file found in any of the expected locations:{bcolors.ENDC}")
    for path in possible_paths:
        print(f"  - {path}")
    return None

def main():
    """
    Main function to run the API key checker tool.
    """
    # [修正] 更新脚本标题
    print(f"{bcolors.HEADER}=== OpenAI-Compatible Gemini Adapter: API Key Checker ==={bcolors.ENDC}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Running as user: {os.getenv('USER', 'unknown')}")
    
    env_file_path = find_env_file()
    if not env_file_path:
        sys.exit(1)

    load_dotenv(dotenv_path=env_file_path)

    api_keys_str = os.getenv("GEMINI_API_KEYS", "")
    initial_keys = [key.strip() for key in api_keys_str.split(',') if key.strip()]

    if not initial_keys or initial_keys == ['your-google-ai-api-key-1', 'your-google-ai-api-key-2']:
        print(f"{bcolors.FAIL}Error: No valid API keys found in {env_file_path}.{bcolors.ENDC}")
        sys.exit(1)

    print(f"\n{bcolors.HEADER}--- Step 1: Pre-processing Keys ---{bcolors.ENDC}")
    print(f"Found {len(initial_keys)} key(s) in .env file.")

    valid_format_keys, invalid_format_keys = [], []
    for key in initial_keys:
        if GEMINI_KEY_PATTERN.match(key):
            valid_format_keys.append(key)
        else:
            invalid_format_keys.append(key)

    if invalid_format_keys:
        print(f"{bcolors.WARNING}Warning: Found {len(invalid_format_keys)} key(s) with an invalid format. They will be ignored:{bcolors.ENDC}")
        for key in invalid_format_keys: print(f"  - {key}")

    unique_keys = list(dict.fromkeys(valid_format_keys))
    if len(valid_format_keys) > len(unique_keys):
        print(f"{bcolors.OKCYAN}Removed {len(valid_format_keys) - len(unique_keys)} duplicate key(s).{bcolors.ENDC}")

    if not unique_keys:
        print(f"{bcolors.FAIL}\nAfter pre-processing, no valid formatted keys remain to be checked. Exiting.{bcolors.ENDC}")
        sys.exit(1)

    print(f"Proceeding to check {len(unique_keys)} unique, valid-format key(s).")
    # [修正] 更新测试的模型名称
    print(f"\n{bcolors.HEADER}--- Step 2: Validating Keys for gemini-1.5-pro-latest (1 sec delay per key) ---{bcolors.ENDC}")

    categorized_keys = {'valid': [], 'temporarily invalid': [], 'permanently invalid': invalid_format_keys}
    for idx, key in enumerate(unique_keys):
        key_display = f"{key[:7]}...{key[-4:]}"
        print(f"[{idx+1}/{len(unique_keys)}] Checking key {bcolors.BOLD}{key_display}{bcolors.ENDC}...", end="", flush=True)
        status, message = check_gemini_api_key(key)
        categorized_keys[status].append(key)
        color_map = {'valid': bcolors.OKGREEN, 'temporarily invalid': bcolors.WARNING, 'permanently invalid': bcolors.FAIL}
        print(f"\r[{idx+1}/{len(unique_keys)}] Key {bcolors.BOLD}{key_display}{bcolors.ENDC}: {color_map[status]}{status.upper()}{bcolors.ENDC} - {message.splitlines()[0]}")
        if idx < len(unique_keys) - 1:
            time.sleep(1)
            
    # [修正] 更新摘要中的模型名称
    print("\n" + "="*22 + " SUMMARY " + "="*22)
    print(f"{bcolors.OKGREEN}Valid keys (can use gemini-1.5-pro-latest): {len(categorized_keys['valid'])}{bcolors.ENDC}")
    print(f"{bcolors.WARNING}Temporarily invalid keys (e.g., out of quota): {len(categorized_keys['temporarily invalid'])}{bcolors.ENDC}")
    print(f"{bcolors.FAIL}Permanently invalid keys: {len(categorized_keys['permanently invalid'])} (includes format-invalid & model access denied){bcolors.ENDC}")
    print("="*53 + "\n")

    try:
        while True:
            print(f"{bcolors.HEADER}--- Step 3: Update .env File ---{bcolors.ENDC}")
            print("1: Keep ONLY the valid keys.")
            print("2: Keep both valid and temporarily invalid keys.")
            print("3: Exit without making any changes.")
            choice = input("Enter your choice (1, 2, or 3): ").strip()
            keys_to_keep = []
            if choice == '1':
                keys_to_keep = categorized_keys['valid']
            elif choice == '2':
                keys_to_keep = categorized_keys['valid'] + categorized_keys['temporarily invalid']
            elif choice == '3':
                print("\nExiting without any changes.")
                sys.exit(0)
            else:
                print(f"{bcolors.FAIL}Invalid choice. Please enter 1, 2, or 3.{bcolors.ENDC}\n")
                continue
            if not keys_to_keep:
                print(f"{bcolors.FAIL}\nError: Your choice would result in zero keys being saved. This is not allowed.{bcolors.ENDC}")
                continue
            update_env_file(keys_to_keep, env_file_path)
            break
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user. Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()
