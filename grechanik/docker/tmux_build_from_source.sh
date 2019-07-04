VERSION=2.8
sudo apt-get -y remove tmux
sudo apt-get -y install wget tar libevent-dev libncurses-dev

mkdir /tmp/tmux-build
cd /tmp/tmux-build
wget https://github.com/tmux/tmux/releases/download/${VERSION}/tmux-${VERSION}.tar.gz
tar xf tmux-${VERSION}.tar.gz
rm -f tmux-${VERSION}.tar.gz
cd tmux-${VERSION}
./configure
make
sudo make install
cd
sudo rm -rf /tmp/tmux-build
