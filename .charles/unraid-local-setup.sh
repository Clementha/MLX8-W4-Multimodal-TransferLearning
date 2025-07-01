# Funther setup for unraid local docker development

# talk to the unraid server, note that key only access not enabled
# ssh -i ~/.ssh/authorized_keys -p 2230 root@192.168.50.243 
ssh -p 2230 root@192.168.50.243 

# create the directory for the workspace
ssh -p 2230 root@192.168.50.243 "mkdir -p /mnt/cache/__tmp/workspace/_github/charles-cai"
# copy the files to the unraid server
scp -P 2230 -r /workspace/_github/ root@192.168.50.243:/mnt/cache/__tmp/workspace/_github/charles-cai
# check the disk space
ssh -p 2230 root@192.168.50.243 "ls -ld /mnt/cache/__tmp/workspace/_github/charles-cai; du -sh /mnt/cache/__tmp/workspace/_github/charles-cai; df -h /mnt/cache"


# copy flaticon data (from MacBook Pro OneDrive folder to unraid server)
scp -r /Users/charles/Library/CloudStorage/OneDrive-Personal/_digital/__media__/flaticon.com/ socialogix-unraid1:/workspace/_github/MLX8-W4-Multimodal-TransferLearning/.charles/.data/


# Run from MacOS to copy the flaticon data to the unraid RAPIDS docker
# scp -r /Users/charles/Library/CloudStorage/OneDrive-Personal/_digital/__media__/flaticon.com/ socialogix-unraid1:/workspace/_github/MLX8-W4-Multimodal-TransferLearning/.charles/.data/

# Example: rsync local MLX8-W4-Multimodal-TransferLearning to unraid server
rsync -avz -e "ssh -p 2230" /workspace/_github/MLX8-W4-Multimodal-TransferLearning/ root@192.168.50.243:/mnt/cache/__tmp/workspace/_github/charles-cai/MLX8-W4-Multimodal-TransferLearning/

# Increase bash history size for Ubuntu
echo 'HISTSIZE=10000' >> ~/.bashrc
echo 'HISTFILESIZE=20000' >> ~/.bashrc
