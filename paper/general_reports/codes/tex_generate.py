import os
import subprocess
import argparse

def generate_latex_from_images(img_dir, feature_dir=None, output_tex='report.tex', images_per_row=3):
    image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))])
    feature_files = []
    if feature_dir:
        # feature_files = sorted([f for f in os.listdir(feature_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))])
        feature_files = sorted(
            [f for f in os.listdir(feature_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))],
            key=lambda x: int(os.path.splitext(x)[0])  # Extract numeric part before extension
        )

    latex_lines = [
        r"\documentclass{article}",
        r"\usepackage{graphicx}",
        r"\usepackage{caption}",
        r"\usepackage{subcaption}",
        r"\usepackage[margin=1in]{geometry}",
        r"\begin{document}",
        r"\section*{Image Report}"
    ]

    # Add main images
    if image_files:
        latex_lines.append(r"\subsection*{Main Images}")
        for i, img in enumerate(image_files):
            if i % images_per_row == 0:
                latex_lines.append(r"\begin{figure}[htbp]")
            latex_lines.append(
                f"""\\begin{{subfigure}}[b]{{{1 / images_per_row - 0.01:.2f}\\textwidth}}
    \\centering
    \\includegraphics[width=\\linewidth]{{{os.path.join(img_dir, img)}}}
    \\caption{{{img}}}
\\end{{subfigure}}"""
            )
            if (i + 1) % images_per_row == 0 or (i + 1) == len(image_files):
                latex_lines.append(r"\end{figure}")
                # latex_lines.append(r"\clearpage")

    # Add feature images
    if feature_files:
        latex_lines.append(r"\subsection*{Feature Images}")
        for i, img in enumerate(feature_files):
            img_path = os.path.join(feature_dir, img)
            escaped_caption = img.replace('_', r'\_')
            if i % images_per_row == 0:
                latex_lines.append(r"\begin{figure}[htbp]")
            latex_lines.append(
                f"""\\begin{{subfigure}}[b]{{{1 / images_per_row - 0.01:.2f}\\textwidth}}
    \\centering
    \\includegraphics[width=\\linewidth]{{\\detokenize{{{os.path.join(feature_dir, img)}}}}}


    \\caption{{{escaped_caption}}}
\\end{{subfigure}}"""
            )
            if (i + 1) % images_per_row == 0 or (i + 1) == len(feature_files):
                latex_lines.append(r"\end{figure}")
                # latex_lines.append(r"\clearpage")

    latex_lines.append(r"\end{document}")
    
    with open(output_tex, 'w') as f:
        f.write('\n'.join(latex_lines))

    print(f"LaTeX file generated: {output_tex}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate LaTeX report from image directories")
    parser.add_argument("--dataset", type=str, required=False, help="Path to the image directory")
    parser.add_argument("--imagedir", type=str, required=True, help="Path to the image directory")
    parser.add_argument("--featuredir", type=str, required=False, help="Path to the feature image directory")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    filename = os.path.basename(args.imagedir)
    last_part = os.path.basename(os.path.normpath(args.imagedir))
    outputfile = args.imagedir + '/'+ last_part + '.tex'
    generate_latex_from_images(args.imagedir, args.featuredir,outputfile, images_per_row=3)
    subprocess.run(["/users/sghosh15/texlive/2025/bin/x86_64-linux/pdflatex", outputfile])
