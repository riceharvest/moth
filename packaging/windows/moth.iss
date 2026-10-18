; moth for Windows: a per-user install (no admin) of the PyInstaller app in dist\moth (packaging/moth.spec).
;   iscc /DAppVersion=0.3.0 /DSourceDir=dist\moth /DIconFile=build\icons\moth.ico /DOutputDir=dist packaging\windows\moth.iss
; Start menu shortcut -> mothw.exe (the first-run window, then the pet). Optional: desktop shortcut, `moth` on PATH,
; start at login (`moth autostart on`: the same Startup entry the first-run window and `moth autostart` manage, so
; there is only ever one). The uninstaller asks whether to run `moth unhook --all` (yes under /SUPPRESSMSGBOXES),
; turns autostart off and stops a running moth first.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\moth"
#endif
#ifndef IconFile
  #define IconFile "..\..\build\icons\moth.ico"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif

[Setup]
AppId={{6B0F3D52-8E4A-4C1B-9B7E-2D5A9C0E7F31}
AppName=moth
AppVersion={#AppVersion}
AppVerName=moth {#AppVersion}
AppPublisher=riceharvest
AppPublisherURL=https://github.com/riceharvest/moth
AppSupportURL=https://github.com/riceharvest/moth/issues
DefaultDirName={autopf}\moth
DefaultGroupName=moth
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=moth-{#AppVersion}-windows-x64-setup
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\mothw.exe
UninstallDisplayName=moth
Compression=lzma2/max
SolidCompression=yes
LZMANumBlockThreads=4
WizardStyle=modern
ChangesEnvironment=yes
CloseApplications=force
RestartApplications=no

[Tasks]
Name: "addtopath"; Description: "Add the moth command to PATH (moth hooks, moth doctor, ... in a new terminal)"
Name: "autostart"; Description: "Start moth when I log in"; Flags: unchecked
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\moth"; Filename: "{app}\mothw.exe"; Comment: "moth, a tiny companion on your screen"
Name: "{autodesktop}\moth"; Filename: "{app}\mothw.exe"; Comment: "moth, a tiny companion on your screen"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; \
  Tasks: addtopath; Check: NotOnPath(ExpandConstant('{app}'))

[Run]
Filename: "{app}\moth.exe"; Parameters: "autostart on"; Flags: runhidden; Tasks: autostart
Filename: "{app}\mothw.exe"; Description: "Start moth"; Flags: postinstall nowait skipifsilent

[Code]
function NotOnPath(Dir: string): Boolean;
var
  Path: string;
begin
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', Path) then
    Path := '';
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Path) + ';') = 0;
end;

procedure StopMoth();
var
  R: Integer;
begin
  { the pet (mothw.exe) and the server it started (mothw.exe -m moth.serve); files in use can't be replaced }
  Exec(ExpandConstant('{sys}\taskkill.exe'), '/F /T /IM mothw.exe', '', SW_HIDE, ewWaitUntilTerminated, R);
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  StopMoth();
  Result := '';
end;

function InitializeUninstall(): Boolean;
var
  R: Integer;
begin
  Result := True;
  if SuppressibleMsgBox('Also remove moth''s hooks from your coding agents (Claude Code, Codex, Hermes, Pi, ' +
                        'OpenCode)?' + #13#10#13#10 + 'This runs: moth unhook --all (it removes only what moth added, ' +
                        'with backups).', mbConfirmation, MB_YESNO, IDYES) = IDYES then
    Exec(ExpandConstant('{app}\moth.exe'), 'unhook --all --yes', '', SW_HIDE, ewWaitUntilTerminated, R);
  Exec(ExpandConstant('{app}\moth.exe'), 'autostart off', '', SW_HIDE, ewWaitUntilTerminated, R);
  StopMoth();
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Path, Dir: string;
  P: Integer;
begin
  if CurUninstallStep <> usPostUninstall then
    exit;
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', Path) then
    exit;
  Dir := ';' + ExpandConstant('{app}');
  P := Pos(Uppercase(Dir), Uppercase(Path));
  if P > 0 then
  begin
    Delete(Path, P, Length(Dir));
    RegWriteExpandStringValue(HKCU, 'Environment', 'Path', Path);
  end;
end;
