DIR=`pwd`
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$DIR:$PYTHONPATH
GAME=cube3
python search_methods/astar.py --states data/$GAME/test/data_0.pkl --model saved_models/$GAME/current/ --env $GAME --weight 0.8 --batch_size 20000 --results_dir results/$GAME/ --language python --nnet_batch_size 10000