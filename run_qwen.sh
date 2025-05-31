OPTION=$1
export CUDA_LAUNCH_BLOCKING=1
export SEED=0

# Llama
############# run large model ##############
# run large, best-of-n, score-method=prm
if [ $OPTION -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=0
    export CONFIG=recipes/Qwen2.5-7B-Instruct/best_of_n.yaml
    python scripts/test_time_compute.py $CONFIG --n=16 --beam_width=1

# run large, beam-search, score-method=prm
elif [ $OPTION -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=1
    export CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search.yaml
    python scripts/test_time_compute.py $CONFIG --n=16 --beam_width=4 

# run large, beam-search, score-method=conf
elif [ $OPTION -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=2
    export CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search.yaml
    python scripts/test_time_compute.py $CONFIG --n=16 --beam_width=4 --score_method=conf

############# run small model ##############
# run small, best-of-n, score-method=prm
elif [ $OPTION -eq 3 ]; then
    export CUDA_VISIBLE_DEVICES=3
    export CONFIG=recipes/Qwen2.5-1.5B-Instruct/best_of_n.yaml
    python scripts/test_time_compute.py $CONFIG --n=16 --beam_width=1

############# run smart ##############
# run smart, best-of-n, score-method=prm
elif [ $OPTION -eq 4 ]; then
    export CUDA_VISIBLE_DEVICES=4
    export CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search_spec.yaml
    python scripts/test_time_compute.py $CONFIG --n=16 --beam_width=1

# run smart, beam-search, score-method=prm
elif [ $OPTION -eq 5 ]; then
    export CUDA_VISIBLE_DEVICES=5
    export CONFIG=recipes/Qwen2.5-7B-Instruct/beam_search_spec.yaml
    python scripts/test_time_compute.py $CONFIG --n=16 --beam_width=4
fi