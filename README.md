# aida-ray-workshop

## Prereqs

Get `uv`:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For other options to install uv see https://docs.astral.sh/uv/getting-started/installation/

## Get started

```
uv run ./hello.py
```

Produces:
```
Hello form aida-ray-workshop!
```

### Activate the venv

To avoid having to prefix things with `uv run`, do:

`source ./.venv/bin/activate`


## Starting ray

Get the IP address of a node that will be the main node:

`ip addr`

Then start ray here (you'll need to activate the venv to have the command):

`ray start --head --node-ip-address 192.168.0.9`

It will show you what to run on other nodes, for me it was:

`ray start --address='192.168.0.9:6379'`
