apt update -q -y
apt install -q -y wget tmux

# Download pre-built Python 3.13 binary (no compilation, ~30 seconds)
wget -q https://github.com/astral-sh/python-build-standalone/releases/download/20260203/cpython-3.13.12+20260203-x86_64-unknown-linux-gnu-install_only.tar.gz -O /tmp/python313.tar.gz
tar -xf /tmp/python313.tar.gz -C /usr/local --strip-components=1
rm /tmp/python313.tar.gz

ln -sf /usr/local/bin/python3.13 /usr/bin/python3
ln -sf /usr/bin/python3 /usr/bin/python
ln -sf /usr/local/bin/pip3.13 /usr/bin/pip3
ln -sf /usr/bin/pip3 /usr/bin/pip

pip install uv
uv pip install --system spoox

python --version
pip --version
tmux -V
pip show spoox