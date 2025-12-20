apt update -q -y
apt install -q -y python3 python3-pip python3-venv
ln -sf /usr/bin/python3 /usr/bin/python
ln -sf /usr/bin/pip3 /usr/bin/pip

apt install -q -y tmux

python3 -m venv /opt/venv
. /opt/venv/bin/activate
pip install spoox

python --version
pip --version
tmux -V
pip show spoox