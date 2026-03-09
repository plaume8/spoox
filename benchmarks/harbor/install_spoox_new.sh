apt update -q -y
apt install -q -y python3 python3-pip python3-venv wget ca-certificates
apt install -q -y tmux

# Download pre-built Python 3.12 binary (no compilation, ~30 seconds)
wget -q https://github.com/indygreg/python-build-standalone/releases/download/20240814/cpython-3.12.5+20240814-x86_64-unknown-linux-gnu-install_only.tar.gz -O /tmp/python312.tar.gz
tar -xf /tmp/python312.tar.gz -C /usr/local --strip-components=1
rm /tmp/python312.tar.gz

ln -sf /usr/local/bin/python3.12 /usr/bin/python3
ln -sf /usr/bin/python3 /usr/bin/python
ln -sf /usr/local/bin/pip3.12 /usr/bin/pip3
ln -sf /usr/bin/pip3 /usr/bin/pip

python3 -m venv /opt/venv
. /opt/venv/bin/activate
pip install spoox

python --version
pip --version
tmux -V
pip show spoox