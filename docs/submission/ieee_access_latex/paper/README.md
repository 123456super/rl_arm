# LaTeX 写作目录

正式投稿 IEEE Access 时，请从 IEEE 官方模板入口下载 IEEE Access LaTeX 模板：

https://ieeeaccess.ieee.org/authors/preparing-your-article/

也可以从 IEEE Author Center 的 Template Selector 进入：

https://template-selector.ieee.org/

## 使用方式

1. 下载 IEEE Access 官方 LaTeX 模板。
2. 将官方模板中的 `ieeeaccess.cls`、`IEEEtran.bst`、示例图和必要样式文件放到本目录。
3. 将 `main.tex` 中的标题、作者、摘要、章节内容替换为正式论文内容。
4. 使用 `latexmk -pdf main.tex` 或 Overleaf 编译。

## 当前目录结构

- `main.tex`：论文主文件骨架。
- `references.bib`：参考文献 BibTeX 占位文件。
- `sections/`：后续可拆分章节。
- `figures/`：论文图片。
- `tables/`：表格源文件或导出的表格。

## 注意

本目录中的 `main.tex` 是为了帮助你尽快开始写作的骨架，不替代 IEEE Access 官方模板。正式投稿前必须以官方模板为准。

