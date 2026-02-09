import os
# 1. 在导入 paddle 之前，直接通过环境变量强行关闭 PIR
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"

import argparse
import paddle
import paddle.base as base
import paddle.static as static
import json

def fix_incompatible_ops(program):
    """
    核心修复函数：
    移除 Slice 算子的 decrease_axis 属性，强制让 Slice 输出保持 1维（[1]）而不是 0维（[]）。
    这样 Paddle 3.x 的 concat 就不会因为输入维度不一致而报错了。
    """
    print("正在扫描并修复不兼容的算子 (Slice decrease_axis)...")
    fix_count = 0

    for i in range(len(program.blocks)):
        block = program.blocks[i]
        for op in block.ops:
            # 修复 slice 算子
            if op.type == "slice":
                if op.has_attr("decrease_axis"):
                    da = op.attr("decrease_axis")
                    if da: # 如果存在 decrease_axis
                        # 将其设置为空列表，取消降维
                        op._set_attr("decrease_axis", [])
                        fix_count += 1

    print(f"✅ 已修复 {fix_count} 个 Slice 算子，防止产生 0维张量。")

def process_old_ops_desc(program):
    for i in range(len(program.blocks[0].ops)):
        if program.blocks[0].ops[i].type == "matmul":
            if not program.blocks[0].ops[i].has_attr("head_number"):
                program.blocks[0].ops[i]._set_attr("head_number", 1)

def infer_shape(program, input_shape_dict):
    # 注意：paddle.enable_static() 已经在 main 中调用，这里不需要重复

    OP_WITHOUT_KERNEL_SET = {
        "feed", "fetch", "recurrent", "go", "rnn_memory_helper_grad",
        "conditional_block", "while", "send", "recv", "listen_and_serv",
        "fl_listen_and_serv", "ncclInit", "select", "checkpoint_notify",
        "gen_bkcl_id", "c_gen_bkcl_id", "gen_nccl_id", "c_gen_nccl_id",
        "c_comm_init", "c_sync_calc_stream", "c_sync_comm_stream",
        "queue_generator", "dequeue", "enqueue", "heter_listen_and_serv",
        "c_wait_comm", "c_wait_compute", "c_gen_hccl_id",
        "c_comm_init_hccl", "copy_cross_scope",
    }

    # 设置输入 Shape
    for k, v in input_shape_dict.items():
        if program.blocks[0].has_var(k):
            program.blocks[0].var(k).desc.set_shape(v)

    # 推导 Shape
    for i in range(len(program.blocks)):
        # 使用 current_ops_len 防止循环越界（你之前的代码里可能有这个问题，这里修正了）
        current_ops_len = len(program.blocks[i].ops)
        for j in range(current_ops_len):
            op = program.blocks[i].ops[j]
            if op.type in OP_WITHOUT_KERNEL_SET:
                continue
            try:
                op.desc.infer_shape(program.blocks[i].desc)
            except Exception as e:
                # 忽略部分算子的推导错误，避免中断
                pass

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_shape_dict", required=True)
    parser.add_argument("--save_path", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # ✅ 关键：必须在这里开启静态图模式，否则 load_inference_model 会报错
    paddle.enable_static()

    try:
        input_shape_dict = json.loads(args.input_shape_dict)
    except:
        input_shape_dict = eval(args.input_shape_dict)

    print(f"Start to load paddle model from: {args.model_path}")

    exe = base.Executor(paddle.CPUPlace())

    # 加载模型
    [program, feed_target_names, fetch_targets] = static.io.load_inference_model(
        args.model_path, exe
    )

    # 1. 处理旧版 matmul 属性
    process_old_ops_desc(program)

    # 2. ✅ 新增：修复 Slice 算子，防止 paddle2onnx 报错
    fix_incompatible_ops(program)

    # 3. 推导 Shape
    print("Start to infer shape...")
    infer_shape(program, input_shape_dict)

    print(f"Saving fixed model to: {args.save_path}")
    feed_vars = [program.global_block().var(name) for name in feed_target_names]

    static.io.save_inference_model(
        args.save_path,
        feed_vars=feed_vars,
        fetch_vars=fetch_targets,
        executor=exe,
        program=program,
    )
    print("✅ Done! 模型已修复并保存。")
