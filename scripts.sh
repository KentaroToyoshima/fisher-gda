#実験本番
#python main.py -mt linear cd leontief -e 20 -b 5 -g 8 -li 0.01 0.01 -cd 0.01 0.01 -Le 0.001 0.001 -mu 1 1 1 -i 10000 -u 500 -a alg2
#python main.py -mt linear cd leontief -e 20 -b 5 -g 8 -li 0.01 0.01 -cd 0.01 0.01 -Le 0.001 0.001 -mu 1 1 1 -i 10000 -u 500 -a m-alg2
#テスト
python main.py -mt linear cd leontief -e 3 -b 5 -g 8 -li 0.01 0.01 -cd 0.01 0.01 -Le 0.001 0.001 -mu 1 1 1 -i 10000 -u 10 -a alg2
python main.py -mt linear cd leontief -e 3 -b 5 -g 8 -li 0.01 0.01 -cd 0.01 0.01 -Le 0.001 0.001 -mu 1 1 1 -i 10000 -u 10 -a m-alg2
python main.py -mt linear cd leontief -e 3 -b 5 -g 8 -li 0.01 0.01 -cd 0.01 0.01 -Le 0.001 0.001 -mu 1 1 1 -i 10000 -u 10 -a alg4