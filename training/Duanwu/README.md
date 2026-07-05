# Duanwu Round 2 Story Model

This is the isolated second-round training workspace for XiaoDie.

All round-2 code, configs, data, reports, logs, and adapter outputs live under `E:\Duanwu`.

Primary goal: improve the Chinese children story model with real, traceable open datasets instead of relying on project-generated synthetic stories.

## Data Sources

- Hugging Face `adam89/TinyStoriesChinese`, license `cdla-sharing-1.0`.
- Hugging Face `opencsg/chinese-cosmopedia`, license `apache-2.0`.
- Global Storybooks Chinese source repository, license `MIT`.
- Project Gutenberg public-domain fairy/folk tale books.

## Commands

```powershell
cd E:\Duanwu
.\scripts\run_pipeline.ps1
```

Run training after the dataset is built:

```powershell
cd E:\Duanwu
.\scripts\run_train_round2.ps1
```

Generate a story:

```powershell
cd E:\Duanwu
.\scripts\env.ps1
python scripts\generate_story.py --keywords "月亮、勇气、分享" --style "睡前安抚" --age "4-5岁"
```
