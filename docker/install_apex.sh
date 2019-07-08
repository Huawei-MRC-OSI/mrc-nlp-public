mkdir /tmp/apex-build
cd /tmp/apex-build
git init
git remote add origin https://github.com/NVIDIA/apex.git
git fetch origin
git reset --hard f29b3f8d3859b8249f15e4835c1f485e4c841ffc
sed -i '50s/raise RuntimeError/print/' setup.py
sudo pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
