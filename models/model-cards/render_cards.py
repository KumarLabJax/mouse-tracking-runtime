"""Script for rendering the model cards."""
from pathlib import Path
import jinja2
import yaml
import argparse
import glob
import os


def render_card_template(template_path, card_data, out_path):
	"""Render a card template with supplied yaml data.

	Args:
		template_path: Jinja2 template file to use for rendering.
		card_data: YAML file to use as fields for rendering.
		out_path: File to write out the rendered template.
	"""
	template = jinja2.Template(Path(template_path).read_text())
	meta = yaml.safe_load(open(card_data, 'r'))
	render = template.render(meta)
	Path(out_path).write_text(render)


def main():
	"""Command line interaction."""
	parser = argparse.ArgumentParser()
	parser.add_argument('--template', help='Jinja2 template for the model cards', default='templates/template.md')
	parser.add_argument('--card-data-folder', help='Folder containing all yaml-formatted card data', default='fields/')
	parser.add_argument('--out-folder', help='Folder where rendered cards are placed', default='cards/')
	args = parser.parse_args()

	files_to_render = glob.glob(args.card_data_folder, recursive=True)
	out_folder = args.out_folder
	if out_folder[-1] != '/':
		out_folder += '/'
	for cur_card_yaml in files_to_render:
		out_file = out_folder + os.path.splitext(os.path.basename(cur_card_yaml))[0]
		render_card_template(args.template, cur_card_yaml, out_file)


if __name__ == '__main__':
	main()
