import subprocess
import sys
import os

# ==================== 实验配置区 ====================
# 每个实验指定：使用哪个训练脚本 + 数据路径 + 参数 + 模型名
EXPERIMENTS = [
###################################################################################################################
    {
        "script": "trainpnoise.py",
        "h5_path": r"D:\zjn\Pnoiuse\JKnet_3001_Pphase_noise.h5",
        "model_name": "EEW_LMBART_pp",
        # "seed": "24"
    },
    
    {
        "script": "trainazi.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LMBERT_azi",
        # "seed": "24"
    },
    {
        "script": "trainazi.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LMBART_azi",
        # "seed": "24"
    },
    {
        "script": "trainazi.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LMChronos_t5_azi",
        # "seed": "42"
    },
    {
        "script": "trainazi.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LM_t5_azi",
        # "seed": "24"
    },
    {
        "script": "trainazi.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LM_GPT2_azi",
        # "seed": "24"
    },


    {
        "script": "trainmag.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LMBERT_mag",
        # "seed": "24"
    },
    {
        "script": "trainmag.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LMBART_mag",
        # "seed": "24"
    },

    {
        "script": "trainmag.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LMChronos_t5_mag",
        # "seed": "24"
    },
    {
        "script": "trainmag.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LM_t5_mag",
        # "seed": "24"
    },

    {
        "script": "trainmag.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LM_GPT2_mag",
        # "seed": "24"
    },

    {
        "script": "traindis.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LMBERT_dis",
        # "seed": "24"
    },


    {
        "script": "traindis.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LMBART_dis",
        # "seed": "24"
    },
    {
        "script": "traindis.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LMChronos_t5_dis",
        # "seed": "24"
    },
    {
        "script": "traindis.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LM_t5_dis",
        # "seed": "24"
    },

    {
        "script": "traindis.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LM_GPT2_dis",
        # "seed": "24"
    },


    {
        "script": "trainmag.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LMBART_magA",
        # "seed": "24"
    },
    {
        "script": "trainmag.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LMBART_magB",
        # "seed": "24"
    },
    {
        "script": "trainmag.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LMBART_magC",
        # "seed": "24"
    },

    {
        "script": "traindis.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LMBART_disA",
        # "seed": "24"
    },
    {
        "script": "traindis.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LMBART_disB",
        # "seed": "24"
    },
    {
        "script": "traindis.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LMBART_disC",
        # "seed": "24"
    },

    {
        "script": "trainazi.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LMBART_aziA",
        # "seed": "24"
    },
    {
        "script": "trainazi.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LMBART_aziB",
        # "seed": "24"
    },    {
        "script": "trainazi.py",
        "h5_path": r"D:\zjn\three\dataset\JKnet_300_with_FS.h5",
        "model_name": "EEW_LMBART_aziC",
        # "seed": "24"
    },

]

# ===================================================

def run_single_experiment(script, h5_path, model_name):
    """运行单次训练实验，支持不同训练脚本"""
    print(f"\n  启动训练 | 脚本: {script} | 模型: {model_name} | 数据: {os.path.basename(h5_path)} | ")

    cmd = [
        sys.executable,
        script,
        "--h5_path", h5_path,
        "--model_name", model_name,

    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))

    result = subprocess.run(cmd, cwd=script_dir)

    if result.returncode != 0:
        print(f"  训练失败: {script} | {model_name} | {h5_path}| ")
        raise RuntimeError(f"Experiment failed for script={script}, model={model_name}")
    else:
        print(f"  训练完成: {script} | {model_name}")


def main():
    total = len(EXPERIMENTS)
    print(f"  共有 {total} 个实验将依次运行...\n")

    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n{'=' * 80}")
        print(f"   实验 {i}/{total}")
        print(f"   脚本: {exp['script']}")
        print(f"   模型: {exp['model_name']}")
        print(f"   数据: {exp['h5_path']}")
        print(f"{'=' * 80}")

        run_single_experiment(
            script=exp["script"],
            h5_path=exp["h5_path"],
            model_name=exp["model_name"],

        )

    print("\n 所有实验已完成！")


if __name__ == "__main__":
    main()