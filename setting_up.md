# CellViT Inference on EPFL RCP Setum

---

## 1. Push the image

```bash
runai submit `                                                                      
>>   --project course-cs-433-group01-[CHANGE TO YOURS] `
>>   --name tissuevit-full `
>>   -i registry.rcp.epfl.ch/jon-docker/tissuevit:v1.1 `
>>   --run-as-uid [CHANGE TO YOURS] `
>>   --run-as-gid [CHANGE TO YOURS] `
>>   -g 1 `
>>   --interactive `                                                                                                                                                                
>>   --existing-pvc claimname=course-cs-433-group01-scratch,path=/data `                                                                                                   
>>   -- sleep infinity 

```



## 2. Connect to the pod and clone the repository 

```bash

kubectl get pods -n runai-course-cs-433-group01-kuci

# After getting the pods connect to bash using the pod name which is usually workload-0-X.

kubectl exec -it -n runai-course-cs-433-group01-kuci tissuevit-full-0-0 -- /bin/bash

cd /data
cd code
```

## 3. Clone the repo if not already there.

Make your own repo there for ease of use.

```bash
mkdir jon[your name]
cd jon

git clone https://<Personal Github Token>@github.com/CS-433/project-2-gradient_tri_scent.git
git config --global --add safe.directory /data/code/project-2-gradient_tri_scent

cd project-2-gradient_tri_scent
```
The token must have repository read and write permission.
Replace `<Personal Github Token>` with your actual GitHub personal access token.

---

## 4. Run a python file

```bash

source /opt/conda/bin/activate tissuevit

python the file_you_want.py

```

## 4. Set up jupyter

In the local bash having activated tissuevit:
You can run this from whatever place you like (I suggest from data so you have the benedict code).

```bash

jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# On another terminal Not inside your server but local:
kubectl port-forward -n runai-course-cs-433-group01-kuci tissuevit-full-0-1 8888:8888

# Register the kernel for jupyter. (Run this inide the server CLI)
source /opt/conda/bin/activate tissuevit
python -m ipykernel install --user --name tissuevit --display-name "tissuevit"

```

# If you want to set up VSCode
Donwside cause you need to do these each time you restart the server.

```bash

cd /data/vscode
./code tunnel

3. Connect from your Laptop
	1. Open VS Code on Windows.
	2. Install the extension: Remote - Tunnels (by Microsoft).
	3. Click the green >< icon (bottom left) -> Connect to Tunnel...
Select GitHub -> Select runai-gpu. [the name that you gave it]


```

# Notebook issue with benedicts code:

Happens when cuda libraries dont match, just run the comands bellow inside tissuevit enviroment.
```bash

pip uninstall -y flash-attn torch torchvision torchaudio xformers
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation --no-cache-dir
pip install xformers --index-url https://download.pytorch.org/whl/cu124


OR

pip uninstall -y flash-attn torch torchvision torchaudio xformers && pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && pip install flash-attn --no-build-isolation --no-cache-dir && pip install xformers --index-url https://download.pytorch.org/whl/cu124


```


