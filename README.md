## Installation
`` conda create -p ./env python=3.8 tensorflow-cpu=2.6 -c conda-forge``
or
`` conda create -p ./env python=3.8 tensorflow=2.6 -c conda-forge -c anaconda``

``conda install requirements.txt -c conda-forge``

``pip install gpflow==2.2.1``
## Trouble Shooting
``pip uninstall dataclasses -y``

## Mounting Erda
Instructions can be found on the DIKU slurm wiki: 
https://diku-dk.github.io/wiki/slurm-cluster#erda
Only the first paragraph is relevant.

Add 
```
Host erda
    HostName io.erda.dk
    User <ku user id: abc123 or ku-email! (in my case)>
```
to `~/.ssh/config`

Be sure to add your public key in ERDA's sftp settings.
