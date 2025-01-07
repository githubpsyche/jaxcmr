# 1) If you don't have 'tree' installed, install it. For example:
#    - On macOS (Homebrew):  brew install tree
#    - On Ubuntu/Debian:     sudo apt-get install tree

# 2) From the root of your homophily project directory:
tree . -L 5 -I "venv|__pycache__|node_modules|.git" > directory_structure.txt

# 3) Optional: Print the structure to the terminal for quick reference:
cat directory_structure.txt
