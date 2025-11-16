import wx
import pyautogui
import json
import math
import threading
from pynput import mouse, keyboard
from pynput.keyboard import Key, GlobalHotKeys

class MacroRecorder(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title='Macro Recorder')
        self.recorded_actions = []
        self.recording = False
        # 最小移动距离阈值，忽略小于此值的抖动
        self.min_move_dist = 5.0
        # 角度变化阈值，方向变化超过此值时记录转折点
        self.angle_threshold_deg = 15.0
        # 热键配置
        self.record_hotkey = 'ctrl+shift+r'
        self.play_hotkey = 'ctrl+shift+p'
        self._hotkey_listener = None
        # 鼠标移动跟踪状态
        self._last_move_point = None
        self._last_move_dir = None
        self._last_recorded_point = None
        # 修饰键状态跟踪
        self._modifier_state = {'ctrl': False, 'shift': False, 'alt': False}
        # 监听器
        self.mouse_listener = None
        self.keyboard_listener = None
        
        self.initialize_ui()
        self.setup_hotkey_listener()
        
    def initialize_ui(self):
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        
        # Control buttons
        self.record_btn = wx.Button(panel, label='Start Recording')
        self.record_btn.Bind(wx.EVT_BUTTON, self.on_record)
        self.play_btn = wx.Button(panel, label='Play')
        self.play_btn.Bind(wx.EVT_BUTTON, self.on_play)
        self.save_btn = wx.Button(panel, label='Save Macro')
        self.save_btn.Bind(wx.EVT_BUTTON, self.on_save)
        self.load_btn = wx.Button(panel, label='Load Macro')
        self.load_btn.Bind(wx.EVT_BUTTON, self.on_load)

        # Hotkey display and config button
        hk_box = wx.BoxSizer(wx.HORIZONTAL)
        self.hotkey_label = wx.StaticText(panel, label=f'Record: {self.record_hotkey}  Play: {self.play_hotkey}')
        self.hk_config_btn = wx.Button(panel, label='Configure Hotkeys')
        self.hk_config_btn.Bind(wx.EVT_BUTTON, self.on_config_hotkeys)
        hk_box.Add(self.hotkey_label, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        hk_box.Add(self.hk_config_btn, 0, wx.ALL, 5)
        
        vbox.Add(self.record_btn, 0, wx.ALL | wx.CENTER, 5)
        vbox.Add(self.play_btn, 0, wx.ALL | wx.CENTER, 5)
        vbox.Add(hk_box, 0, wx.ALL | wx.EXPAND, 5)
        vbox.Add(self.save_btn, 0, wx.ALL | wx.CENTER, 5)
        vbox.Add(self.load_btn, 0, wx.ALL | wx.CENTER, 5)
        
        panel.SetSizer(vbox)
        self.SetSize(800, 800)
        
    def on_record(self, event):
        if not self.recording:
            self.recording = True
            self.record_btn.SetLabel('Stop Recording')
            self.recorded_actions = []
            self.start_recording()
        else:
            self.recording = False
            self.record_btn.SetLabel('Start Recording')
            self.stop_recording()

    def on_play(self, event):
        """回放录制的宏"""
        if not self.recorded_actions:
            return
        
        # 使用 pyautogui 的 PAUSE 设置优化性能
        pyautogui.PAUSE = 0.01
        
        for action in self.recorded_actions:
            if action['type'] == 'mouse':
                pyautogui.moveTo(action['x'], action['y'], duration=0.01)
                if action.get('click'):
                    pyautogui.click()
            elif action['type'] == 'keyboard':
                key = action.get('key')
                modifiers = action.get('modifiers', [])
                
                # 按下修饰键
                for mod in modifiers:
                    pyautogui.keyDown(mod)
                
                # 按下主键
                if isinstance(key, str) and key.startswith('Key.'):
                    kname = key.split('.', 1)[1]
                    try:
                        pyautogui.press(kname)
                    except Exception:
                        pass
                else:
                    pyautogui.press(key)
                
                # 释放修饰键
                for mod in reversed(modifiers):
                    pyautogui.keyUp(mod)

    def on_save(self, event):
        with wx.FileDialog(self, "Save Macro File", wildcard="JSON files (*.json)|*.json",
                          style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            pathname = fileDialog.GetPath()
            try:
                with open(pathname, 'w') as file:
                    json.dump(self.recorded_actions, file)
            except IOError:
                wx.LogError(f"Cannot save current data in file '{pathname}'.")
                
    def on_load(self, event):
        with wx.FileDialog(self, "Open Macro File", wildcard="JSON files (*.json)|*.json",
                          style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            pathname = fileDialog.GetPath()
            try:
                with open(pathname, 'r') as file:
                    self.recorded_actions = json.load(file)
            except IOError:
                wx.LogError(f"Cannot open file '{pathname}'.")
                
    def _format_hotkey_for_pynput(self, hk: str):
        """将用户可读的热键字符串转换为 pynput GlobalHotKeys 格式"""
        mapping_mod = {'ctrl': 'ctrl', 'alt': 'alt', 'shift': 'shift', 'cmd': 'cmd', 'win': 'cmd', 'super': 'cmd'}
        mapping_special = {'space': 'space', 'enter': 'enter', 'tab': 'tab', 'esc': 'esc', 'escape': 'esc', 'backspace': 'backspace'}
        parts = [p.strip().lower() for p in hk.split('+') if p.strip()]
        out_parts = []
        for p in parts:
            if p in mapping_mod:
                out_parts.append(f'<{mapping_mod[p]}>')
            elif p in mapping_special:
                out_parts.append(f'<{mapping_special[p]}>')
            else:
                out_parts.append(p)
        return '+'.join(out_parts)

    def setup_hotkey_listener(self):
        """设置全局热键监听"""
        try:
            if self._hotkey_listener:
                self._hotkey_listener.stop()
        except Exception:
            pass

        try:
            rk = self._format_hotkey_for_pynput(self.record_hotkey)
            pk = self._format_hotkey_for_pynput(self.play_hotkey)
            hotkeys = {
                rk: lambda: wx.CallAfter(self.on_record, None),
                pk: lambda: wx.CallAfter(self.on_play, None)
            }
            self._hotkey_listener = GlobalHotKeys(hotkeys)
            self._hotkey_listener.start()
        except Exception as e:
            print('Hotkey listener failed:', e)
            self._hotkey_listener = None

    def on_config_hotkeys(self, event):
        """配置热键对话框"""
        def capture_hotkey_dialog(title, default_value):
            dlg = wx.Dialog(self, title=title, style=wx.DEFAULT_DIALOG_STYLE)
            sizer = wx.BoxSizer(wx.VERTICAL)
            info = wx.StaticText(dlg, label='Press desired hotkey combination (supports multiple modifiers). Press Esc to cancel.')
            current = wx.StaticText(dlg, label=f'Current: {default_value}')
            cancel_btn = wx.Button(dlg, wx.ID_CANCEL, label='Cancel')
            sizer.Add(info, 0, wx.ALL, 10)
            sizer.Add(current, 0, wx.ALL, 10)
            sizer.Add(cancel_btn, 0, wx.ALL | wx.CENTER, 5)
            dlg.SetSizerAndFit(sizer)

            captured = {'hotkey': None}
            stop_event = threading.Event()
            mods = set()

            def is_modifier_key(k):
                return k in (Key.ctrl_l, Key.ctrl_r, Key.shift_l, Key.shift_r, Key.alt_l, Key.alt_r)

            def mod_name_from_key(k):
                mod_map = {
                    (Key.ctrl_l, Key.ctrl_r): 'ctrl',
                    (Key.shift_l, Key.shift_r): 'shift',
                    (Key.alt_l, Key.alt_r): 'alt'
                }
                for keys, name in mod_map.items():
                    if k in keys:
                        return name
                return None

            def key_to_name(k):
                if hasattr(k, 'char') and k.char is not None:
                    return k.char.lower()
                s = str(k)
                return s.split('.', 1)[1] if s.startswith('Key.') else s

            def on_press(k):
                if stop_event.is_set():
                    return
                if is_modifier_key(k):
                    mn = mod_name_from_key(k)
                    if mn:
                        mods.add(mn)
                    wx.CallAfter(current.SetLabel, f'Current: {" + ".join(sorted(mods))}')
                    return
                if k == Key.esc:
                    captured['hotkey'] = None
                    stop_event.set()
                    return
                main = key_to_name(k)
                parts = [m for m in ['ctrl', 'shift', 'alt'] if m in mods] + [main]
                captured['hotkey'] = '+'.join(parts)
                wx.CallAfter(current.SetLabel, f'Captured: {captured["hotkey"]}')
                stop_event.set()

            def on_release(k):
                if is_modifier_key(k):
                    mn = mod_name_from_key(k)
                    if mn and mn in mods:
                        mods.discard(mn)
                    wx.CallAfter(current.SetLabel, f'Current: {" + ".join(sorted(mods))}')

            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()

            timer = wx.Timer(dlg)
            def on_timer(evt):
                if stop_event.is_set():
                    timer.Stop()
                    try:
                        listener.stop()
                    except Exception:
                        pass
                    dlg.EndModal(wx.ID_OK)
            dlg.Bind(wx.EVT_TIMER, on_timer)
            timer.Start(50)

            res = dlg.ShowModal()
            if res == wx.ID_CANCEL:
                try:
                    listener.stop()
                except Exception:
                    pass
            timer.Stop()
            dlg.Destroy()
            return captured['hotkey'] if res == wx.ID_OK else None

        new_r = capture_hotkey_dialog('Configure Record Hotkey', self.record_hotkey)
        if new_r:
            self.record_hotkey = new_r

        new_p = capture_hotkey_dialog('Configure Play Hotkey', self.play_hotkey)
        if new_p:
            self.play_hotkey = new_p

        self.hotkey_label.SetLabel(f'Record: {self.record_hotkey}  Play: {self.play_hotkey}')
        self.setup_hotkey_listener()

    def start_recording(self):
        """开始录制宏"""
        def distance(a, b):
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            return math.hypot(dx, dy)

        def normalize(v):
            d = math.hypot(v[0], v[1])
            return (v[0]/d, v[1]/d) if d > 0 else (0.0, 0.0)

        def angle_between(u, v):
            """计算两个归一化向量之间的角度"""
            dot = max(-1.0, min(1.0, u[0]*v[0] + u[1]*v[1]))
            return math.degrees(math.acos(dot))

        def record_point(pt):
            """记录鼠标点"""
            self.recorded_actions.append({
                'type': 'mouse',
                'x': int(pt[0]),
                'y': int(pt[1])
            })
            self._last_recorded_point = pt

        def on_move(x, y):
            """鼠标移动事件处理"""
            if not self.recording:
                return
            pt = (x, y)
            if self._last_move_point is None:
                self._last_move_point = pt
                self._last_move_dir = None
                record_point(pt)
                return

            dist = distance(pt, self._last_move_point)
            if dist < self.min_move_dist:
                self._last_move_point = pt
                return

            cur_dir = normalize((pt[0] - self._last_move_point[0], pt[1] - self._last_move_point[1]))
            if self._last_move_dir is None:
                self._last_move_dir = cur_dir
                self._last_move_point = pt
                return

            ang = angle_between(self._last_move_dir, cur_dir)
            if ang >= self.angle_threshold_deg:
                prev_pt = self._last_move_point
                if self._last_recorded_point is None or distance(prev_pt, self._last_recorded_point) >= self.min_move_dist:
                    record_point(prev_pt)
                self._last_move_dir = cur_dir

            self._last_move_point = pt

        def on_click(x, y, button, pressed):
            """鼠标点击事件处理"""
            if self.recording and pressed:
                self.recorded_actions.append({
                    'type': 'mouse',
                    'x': int(x),
                    'y': int(y),
                    'click': True
                })

        def on_press(key):
            """键盘按下事件处理"""
            if not self.recording:
                return
                
            # 更新修饰键状态
            mod_map = {
                (Key.ctrl_l, Key.ctrl_r): 'ctrl',
                (Key.shift_l, Key.shift_r): 'shift',
                (Key.alt_l, Key.alt_r): 'alt'
            }
            for keys, mod_name in mod_map.items():
                if key in keys:
                    self._modifier_state[mod_name] = True
                    break
            
            active_modifiers = [mod for mod, active in self._modifier_state.items() if active]
            
            try:
                if hasattr(key, 'char') and key.char:
                    self.recorded_actions.append({
                        'type': 'keyboard',
                        'key': key.char,
                        'modifiers': active_modifiers
                    })
                else:
                    self.recorded_actions.append({
                        'type': 'keyboard',
                        'key': str(key),
                        'modifiers': active_modifiers
                    })
            except AttributeError:
                pass

        def on_release(key):
            """键盘释放事件处理"""
            if not self.recording:
                return
                
            mod_map = {
                (Key.ctrl_l, Key.ctrl_r): 'ctrl',
                (Key.shift_l, Key.shift_r): 'shift',
                (Key.alt_l, Key.alt_r): 'alt'
            }
            for keys, mod_name in mod_map.items():
                if key in keys:
                    self._modifier_state[mod_name] = False
                    break

        # 重置状态
        self._last_move_point = None
        self._last_move_dir = None
        self._last_recorded_point = None

        self.mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
        self.keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.mouse_listener.start()
        self.keyboard_listener.start()

    def stop_recording(self):
        """停止录制宏"""
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        self._modifier_state = {'ctrl': False, 'shift': False, 'alt': False}

if __name__ == '__main__':
    app = wx.App()
    frame = MacroRecorder()
    frame.Show()
    app.MainLoop()
