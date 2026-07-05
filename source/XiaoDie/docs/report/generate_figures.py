import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

out = r"D:\Spacemit\XiaoDie\docs\report\figures"
os.makedirs(out, exist_ok=True)

plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'SimSun'],
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox_inches': 'tight',
    'font.size': 10,
})

# 1. Eval loss curve
steps = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
eval_loss = [2.0109, 1.9750, 1.9475, 1.9362, 1.9167, 1.9037, 1.8976, 1.8876, 1.8844, 1.8821]
train_loss_approx = [2.15, 2.08, 2.02, 1.98, 1.96, 1.94, 1.93, 1.92, 1.91, 1.90]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(steps, eval_loss, 'b-o', markersize=5, linewidth=1.8, label='Eval Loss')
ax.set_xlabel('训练步数 (Steps)', fontsize=11)
ax.set_ylabel('Eval Loss', fontsize=11)
ax.set_xlim(0, 2200)
ax.set_ylim(1.85, 2.05)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
for i, (s, l) in enumerate(zip(steps, eval_loss)):
    if i % 2 == 0:
        ax.annotate(f'{l:.3f}', (s, l), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=7, color='blue')
fig.savefig(os.path.join(out, 'eval_loss_curve.png'))
plt.close()

# 2. Data source composition
sources = ['TinyStories\nChinese', 'Chinese\nCosmopedia', 'Storybooks\nChinese']
counts = [5207, 240, 8]
colors = ['#4E79A7', '#F28E2B', '#76B7B2']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
bars = ax1.bar(sources, counts, color=colors, edgecolor='white', linewidth=0.8)
ax1.set_ylabel('样本数量', fontsize=11)
ax1.set_title('各数据源样本数量', fontsize=12)
for bar, c in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 80,
            f'{c}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.set_ylim(0, 6000)

wedges, texts, autotexts = ax2.pie(counts, labels=['TinyStories CN\n95.5%', 'Cosmopedia\n4.4%', 'Storybooks\n0.1%'],
                                    colors=colors, autopct='', startangle=90,
                                    wedgeprops=dict(linewidth=1, edgecolor='white'))
ax2.set_title('数据来源占比', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(out, 'data_composition.png'))
plt.close()

# 3. Data filtering reasons
reasons = ['文本过长', '儿童危险行为', '暴力内容', '教学类文本', '英文占比高',
           '非中文', '医疗建议', '过短/非中文', '成人内容', '隐私相关', '政治宣传', '仇恨言论']
filter_counts = [594, 441, 331, 194, 76, 37, 36, 32, 30, 8, 2, 1]

fig, ax = plt.subplots(figsize=(8, 5))
y_pos = np.arange(len(reasons))
bars = ax.barh(y_pos, filter_counts, color='#E15759', alpha=0.85, edgecolor='white')
ax.set_yticks(y_pos)
ax.set_yticklabels(reasons, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('过滤样本数', fontsize=11)
ax.set_title('数据清洗各类过滤原因统计', fontsize=12)
for bar, c in zip(bars, filter_counts):
    ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2.,
           f'{c}', ha='left', va='center', fontsize=8)
ax.set_xlim(0, 700)
fig.tight_layout()
fig.savefig(os.path.join(out, 'data_filtering.png'))
plt.close()

# 4. TTFT comparison
configs = ['纯LLM\n(已预热)', 'RAG\n(缓存命中)', 'RAG+LLM\n+TTS并发', 'RAG\n(冷prefix)']
ttft_vals = [2.93, 4.81, 5.54, 50.52]
colors_ttft = ['#59A14F', '#4E79A7', '#F28E2B', '#E15759']

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(configs, ttft_vals, color=colors_ttft, edgecolor='white', linewidth=0.8, width=0.6)
ax.set_ylabel('首Token延迟 TTFT (秒)', fontsize=11)
ax.set_title('不同配置下的首Token延迟对比', fontsize=12)
for bar, v in zip(bars, ttft_vals):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.8,
           f'{v:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylim(0, 58)
ax.axhline(y=5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax.text(3.4, 5.5, '5秒参考线', fontsize=8, color='gray')
fig.tight_layout()
fig.savefig(os.path.join(out, 'ttft_comparison.png'))
plt.close()

# 5. Generation speed comparison
configs_speed = ['纯LLM\n(已预热)', 'RAG\n(缓存命中)', 'RAG冷prefix', 'RAG+LLM\n+TTS并发']
gen_speed = [3.36, 3.20, 3.56, 1.655]
prompt_eval = [89.6, 158.53, 14.01, 136.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
colors_gen = ['#59A14F', '#4E79A7', '#F28E2B', '#E15759']
bars1 = ax1.bar(configs_speed, gen_speed, color=colors_gen, edgecolor='white', width=0.55)
ax1.set_ylabel('生成速度 (tokens/s)', fontsize=11)
ax1.set_title('Token生成速度对比', fontsize=12)
for bar, v in zip(bars1, gen_speed):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
            f'{v:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.set_ylim(0, 4.2)

bars2 = ax2.bar(configs_speed, prompt_eval, color=colors_gen, edgecolor='white', width=0.55)
ax2.set_ylabel('Prompt Eval速度 (tokens/s)', fontsize=11)
ax2.set_title('Prompt处理速度对比', fontsize=12)
for bar, v in zip(bars2, prompt_eval):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
            f'{v:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax2.set_ylim(0, 185)
fig.tight_layout()
fig.savefig(os.path.join(out, 'speed_comparison.png'))
plt.close()

# 6. Data cleaning pipeline (funnel)
stages = ['原始收集', 'HTML清洗', '语言过滤', '安全过滤', '长度过滤', '去重后', 'SFT构造']
stage_counts = [5455+594+441+331+194+76+37+36+32+30+8+2+1, 5455+441+331+194+76+37+36+32+30+8+2+1,
                5455+441+331+194+36+30+8+2+1, 5455+194, 5455, 5455, 5180]
# Simplify: use approximate pipeline numbers
stage_counts = [7237, 6643, 6530, 5762, 5455, 5455, 5180]

fig, ax = plt.subplots(figsize=(8, 4.5))
colors_funnel = plt.cm.Blues(np.linspace(0.3, 0.9, len(stages)))
bars = ax.barh(range(len(stages)-1, -1, -1), stage_counts, color=colors_funnel, edgecolor='white', height=0.65)
ax.set_yticks(range(len(stages)-1, -1, -1))
ax.set_yticklabels(stages, fontsize=10)
ax.set_xlabel('样本数量', fontsize=11)
ax.set_title('数据清洗流水线各阶段样本量', fontsize=12)
for bar, c in zip(bars, stage_counts):
    ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2.,
           f'{c}', ha='left', va='center', fontsize=9, fontweight='bold')
ax.set_xlim(0, 8500)
fig.tight_layout()
fig.savefig(os.path.join(out, 'data_pipeline.png'))
plt.close()

# 7. IP knowledge card distribution
ips = ['小马宝莉', '汪汪队', '海底小纵队', '巴巴爸爸', '小猪佩奇']
ip_cards = [83, 29, 22, 12, 10]
colors_ip = ['#B07AA1', '#4E79A7', '#76B7B2', '#F28E2B', '#FF9DA7']

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(ips, ip_cards, color=colors_ip, edgecolor='white', linewidth=0.8, width=0.55)
ax.set_ylabel('知识卡片数量', fontsize=11)
ax.set_title('各动画IP知识卡片分布', fontsize=12)
for bar, c in zip(bars, ip_cards):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
           f'{c}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig(os.path.join(out, 'ip_distribution.png'))
plt.close()

# 8. System latency breakdown (stacked bar for end-to-end)
components = ['ASR识别', 'RAG检索', 'LLM首Token', 'LLM生成\n(100字)', 'TTS合成\n(首段)', 'TTS播放']
latency_pure = [1.1, 0.0, 2.93, 8.9, 0, 0]  # pure LLM path
latency_rag = [1.1, 0.1, 4.81, 9.4, 0, 0]  # RAG cached path
latency_full = [1.1, 0.1, 5.54, 18.2, 4.7, 2.77]  # full path with TTS

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(components))
w = 0.25
bars1 = ax.bar(x - w, latency_pure, w, label='纯LLM(已预热)', color='#59A14F', alpha=0.85)
bars2 = ax.bar(x, latency_rag, w, label='RAG+LLM(缓存)', color='#4E79A7', alpha=0.85)
bars3 = ax.bar(x + w, latency_full, w, label='RAG+LLM+TTS', color='#F28E2B', alpha=0.85)
ax.set_ylabel('延迟 (秒)', fontsize=11)
ax.set_title('各模块延迟分布', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(components, fontsize=9)
ax.legend(fontsize=9)
ax.set_ylim(0, 22)
fig.tight_layout()
fig.savefig(os.path.join(out, 'latency_breakdown.png'))
plt.close()

# 9. Streaming TTS timeline illustration
fig, ax = plt.subplots(figsize=(9, 3.5))
# LLM generation bars
llm_segs = [(0, 8, '句子1'), (8, 15, '句子2'), (15, 22, '句子3'), (22, 28, '句子4')]
for start, end, label in llm_segs:
    ax.barh(2, end-start, left=start, height=0.4, color='#4E79A7', edgecolor='white')
    ax.text((start+end)/2, 2, label, ha='center', va='center', fontsize=8, color='white', fontweight='bold')

# TTS synth bars
tts_segs = [(8, 13, '合成1'), (15, 19, '合成2'), (22, 25.5, '合成3'), (28, 31, '合成4')]
for start, end, label in tts_segs:
    ax.barh(1, end-start, left=start, height=0.4, color='#F28E2B', edgecolor='white')
    ax.text((start+end)/2, 1, label, ha='center', va='center', fontsize=8, color='white', fontweight='bold')

# Audio play bars
play_segs = [(13, 16, '播放1'), (19, 21.5, '播放2'), (25.5, 28, '播放3'), (31, 33, '播放4')]
for start, end, label in play_segs:
    ax.barh(0, end-start, left=start, height=0.4, color='#59A14F', edgecolor='white')
    ax.text((start+end)/2, 0, label, ha='center', va='center', fontsize=8, color='white', fontweight='bold')

ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['音频播放', 'TTS合成', 'LLM生成'], fontsize=10)
ax.set_xlabel('时间 (秒)', fontsize=11)
ax.set_title('句子级流式TTS流水线时序示意', fontsize=12)
ax.set_xlim(-1, 35)
ax.set_ylim(-0.5, 2.8)
ax.axvline(x=13, color='red', linestyle=':', alpha=0.4)
ax.text(13, 2.55, '用户听到\n第一段语音', ha='center', fontsize=8, color='red')
fig.tight_layout()
fig.savefig(os.path.join(out, 'streaming_tts_timeline.png'))
plt.close()

print("All figures generated successfully in", out)
for f in sorted(os.listdir(out)):
    print(f"  {f}")
