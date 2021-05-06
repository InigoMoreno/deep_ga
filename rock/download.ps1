$REMOTE='https://robotics.estec.esa.int/';
$REMOTEFOLDER = 'owncloud/public.php/webdav/';
$USERNAME = 'vc96L9nGPW8NWjP';
$PASSWORD = Read-Host -Prompt 'Input the password';

# Define custom auth header
$secpasswd = ConvertTo-SecureString $PASSWORD -AsPlainText -Force
$cred = New-Object System.Management.Automation.PSCredential($USERNAME, $secpasswd)
$authH = @{Authorization='Basic {0}' -f [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes(("${USERNAME}:${PASSWORD}")))};

# Define custom WebRequest function to allow WebDav-Methods PROPFIND and MKCOL
# With PowerShell 6 use "Invoke-RestMethod -CustomMethod" instead
function Invoke-CustomRequest($Method, $Url){
    $req = [net.webrequest]::create($Url);
    $req.Headers['Authorization'] = $authH.Authorization;
    $req.Headers['Depth'] = '1';
    $req.Method = $Method;
    $reader = [System.IO.StreamReader]::new($req.GetResponse().GetResponseStream());
    $props = $reader.ReadToEnd();
    return $props;
}
function Get-Webdav-ls($Path = "", $Match = ""){
    $dirs = Invoke-CustomRequest -Method 'PROPFIND' -Url $REMOTE$REMOTEFOLDER$Path
    $dirs = $dirs | Select-Xml -XPath "//*[local-name()='href']"
    $dirs = $dirs | ForEach-Object { $_.Node.'#text'.substring("$REMOTEFOLDER$Path".Length+1)}
    $dirs = $dirs | Where-Object{$_ -match $Match}
    return $dirs;
}
function Get-Webdav-Size($Path){
    $req = Invoke-WebRequest -Uri $REMOTE$REMOTEFOLDER$Path -Credential $cred -Method Head
    $length = $req.Headers.'Content-Length'
    return ($length[0] -as [long])
}

function Invoke-Webdav-Download($Path){
    echo $REMOTE$REMOTEFOLDER$Path
    mkdir -Force (Split-Path $Path) | Out-Null
    Invoke-WebRequest  -SkipCertificateCheck -Uri $REMOTE$REMOTEFOLDER$Path -Credential $cred -o $Path -Resume | Out-Null
}

# echo "Computing total size"
# $totalsize = 0
# Foreach ($day in Get-Webdav-ls -Match "June/"){
#     Foreach ($folder in Get-Webdav-ls -Path "$($day)Traverse" -Match "201706"){
#         Foreach ($log in @("bb3.log","waypoint_navigation.log","imu.log")){
#             $logsize = Get-Webdav-Size -Path "$($day)Traverse$folder$log"
#             $totalsize = $totalsize + $logsize/1073741824
#         }
#     }
# }
$totalsize = 719.993387400173 
echo "TotalSize: $([math]::Round($totalsize,1)) GB"
echo "Starting Download"

$ply_file = Get-Webdav-ls -Path "Maps" -Match "ply"

Invoke-Webdav-Download "Maps$ply_file"

$downloadedsize = 0
Foreach ($day in Get-Webdav-ls -Match "June"){
    Foreach ($folder in Get-Webdav-ls -Path "$($day)Traverse" -Match "201706"){
        Foreach ($log in @("imu.log","bb3.log","waypoint_navigation.log")){
            $path = "$($day)Traverse$folder$log"
            $ProgressPreference = 'Continue'
            $logsize = Get-Webdav-Size $path
            Write-Progress -Activity "Downloading logs" `
                           -status "Downloadin file of length: $logsize ($($day)Traverse$folder$log) /n"`
                           -percentComplete ($downloadedSize / $totalsize*100)
            # $ProgressPreference = 'SilentlyContinue'
            echo $path
            Invoke-Webdav-Download $path
            $downloadedsize = $downloadedsize + $logsize/1073741824
            echo $downloadedSize
        }
    }
}


# md -Force "$($day)Traverse$folder"
# Invoke-WebRequest -Uri $path -Credential $cred -o "$($day)Traverse$folder$log" -Resume