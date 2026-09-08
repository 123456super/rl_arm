# IEEE Access 官方 LaTeX 模板下载与放置方法

整理日期：2026-09-08

## 方法一：浏览器下载

1. 打开 IEEE Access 的官方准备页面：

   https://ieeeaccess.ieee.org/authors/preparing-your-article/

2. 找到 `Article Acceptance Requirements` 下面的模板下载区域。

3. 在 `LaTex` 下面点击 `Download Template`。

4. 目前官网对应的直接下载链接是：

   https://ieeeaccess.ieee.org/wp-content/uploads/2026/05/ACCESS_latex_template_20260513-1-1.zip

5. 下载后解压这个 zip 文件。

6. 把解压出来的模板文件复制到本项目目录：

   ```text
   /home/c211/jxf/rl/docs/submission/ieee_access_latex/paper/
   ```

   重点需要有：

   - `ieeeaccess.cls`
   - `IEEEtran.bst` 或模板自带的参考文献样式文件
   - 官方示例 `.tex`
   - 官方示例图片或样式资源

7. 然后用本资料包中的 `main.tex` 作为写作骨架，或者把其中的标题、摘要、章节内容复制到官方示例 `.tex` 中。

## 方法二：命令行下载

在项目根目录运行：

```bash
cd /home/c211/jxf/rl/docs/submission/ieee_access_latex/paper
wget https://ieeeaccess.ieee.org/wp-content/uploads/2026/05/ACCESS_latex_template_20260513-1-1.zip
unzip ACCESS_latex_template_20260513-1-1.zip
```

如果没有 `wget`，可以用：

```bash
cd /home/c211/jxf/rl/docs/submission/ieee_access_latex/paper
curl -L -O https://ieeeaccess.ieee.org/wp-content/uploads/2026/05/ACCESS_latex_template_20260513-1-1.zip
unzip ACCESS_latex_template_20260513-1-1.zip
```

解压后建议检查：

```bash
find . -maxdepth 2 -type f | sort
```

看到 `ieeeaccess.cls` 之后，说明模板类文件已经在本地了。

## 方法三：Overleaf

如果你用 Overleaf：

1. 下载 IEEE Access 官方 LaTeX zip 模板。
2. 在 Overleaf 新建项目。
3. 上传整个 zip。
4. IEEE 官方特别提醒：如果用 Overleaf，要在 Overleaf 里面解压模板文件。
5. 再把本项目 `paper/main.tex` 里的论文内容复制到 Overleaf 的主 `.tex` 文件里。

## 编译方式

如果本地安装了 LaTeX，可以在 `paper/` 目录运行：

```bash
latexmk -pdf main.tex
```

如果没有 `latexmk`，可以尝试：

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

正式投稿前还要用 IEEE 工具检查：

- IEEE PDF Checker: https://ieee-pdfchecker.org/
- IEEE LaTeX Analyzer: https://latexqc.ieee.org/

## 建议做法

最稳的做法是：先下载官方模板，确认官方示例能编译，再把本资料包 `main.tex` 中的论文结构迁移进去。

不要随便使用第三方 GitHub 上的 IEEE Access 模板，因为 IEEE Access 的模板文件会更新，投稿系统通常以官网模板为准。
