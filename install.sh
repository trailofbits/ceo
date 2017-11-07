rm -Rf tools

mkdir -p tools/grr
cd tools/grr
git clone https://github.com/ggrieco-tob/grr src
cd src
make all install GRANARY_TARGET=release GRANARY_PREFIX_DIR=..

cd ../../..

mkdir -p ceo/tools/grr
cp -R tools/grr/bin ceo/tools/grr

mkdir -p tools/afl-cgc
cd tools/afl-cgc
git clone https://github.com/ggrieco-tob/afl-cgc src
cd src
make
cd qemu_mode
./build_qemu_support.sh
cd ..
make install PREFIX=..

cd ../../..

mkdir -p ceo/tools/afl-cgc
cp -R tools/afl-cgc/bin ceo/tools/afl-cgc

mkdir -p tools/manticore
cd tools/manticore
git clone https://github.com/ggrieco-tob/manticore src
cd src
python2 setup.py install --user

cd ../../..

python2 setup.py install --user
