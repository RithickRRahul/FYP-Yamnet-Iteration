$git = "C:\Program Files\Git\cmd\git.exe"
& $git config --global user.email "rithick.r.rahul@gmail.com"
& $git config --global user.name "RithickRRahul"
& $git init
& $git add .
& $git commit -m "Initialize Pretrained Baseline for Multimodal Violence Detection"
& $git branch -M main
& $git remote add origin https://github.com/RithickRRahul/FYP-Yamnet-Iteration.git
& $git push -u origin main
