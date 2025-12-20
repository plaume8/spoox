apt update -q -y
apt install -q -y python3 python3-pip python3-venv
ln -sf /usr/bin/python3 /usr/bin/python
ln -sf /usr/bin/pip3 /usr/bin/pip

# ensure python >= 3.10 is installed
PYV=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
if dpkg --compare-versions "$PYV" lt "3.10"; then
    echo "Python < 3.10 detected. Installing Python 3.10..."

    apt install -q -y software-properties-common

    add-apt-repository 'deb http://deb.pascalroeleven.nl/python3.10 bullseye-backports main'
    wget -qO- https://pascalroeleven.nl/deb-pascalroeleven.gpg | tee /etc/apt/trusted.gpg.d/python3.10-backport.gpg

    apt update -q -y

    apt install -q -y python3.10  python3.10-venv

    ln -sf /usr/bin/python3.10 /usr/bin/python3
    ln -sf /usr/bin/python3.10 /usr/bin/python
fi

# --

apt install -q -y tmux

python3.10 -m venv /opt/venv  # todo revert to python 3 only
. /opt/venv/bin/activate
pip install spoox

python --version
pip --version
tmux -V
pip show spoox