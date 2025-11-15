set -x

# MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"  # replace it with your local file path
MODEL_PATH=/path/to/IE-Critic-CoT # your own cot + direct sft model path

python3 -m verl.trainer.main \
    config=examples/config_ie_critic_r1.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.offload.offload_optimizer=True \
    worker.rollout.limit_images=2 \
    worker.reward.score_function="./examples/score_function/ie_critic_r1.py:compute_score_l1" \
    trainer.save_freq=-1 \
    trainer.experiment_name=ie_critic_r1 \
    trainer.save_checkpoint_path="./saves/ie_critic_r1" \
    trainer.total_episodes=5 \
    trainer.logger=['console','wandb']