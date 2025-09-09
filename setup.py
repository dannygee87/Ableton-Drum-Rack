from setuptools import setup

setup(
	name='abletondrumrack',
	version='1.1.0',
	author='Daniel Grimm',
	author_email='',
	description='Create Ableton Live 12 Drum Rack presets.',
	url='https://github.com/danikadannygee87/Ableton-Drum-Rack',
	license='MIT',
	packages=['.abletondrumrack'],
	package_data={'.abletondrumrack': ['ableton_files/*']},
	install_requires=[
		'numpy',
		'pandas',
		'SoundFile',
		],
	python_requires='>=3.7',
)
