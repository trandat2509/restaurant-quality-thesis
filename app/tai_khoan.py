"""Đăng ký, đăng nhập và lưu tài khoản của ứng dụng trên máy chủ."""
from contextlib import closing
from hashlib import scrypt
from hmac import compare_digest
import os
from pathlib import Path
import re
from secrets import token_bytes
import sqlite3
from time import time

import streamlit as st


MIN_PASSWORD = 15
MAX_PASSWORD = 128
MAX_ATTEMPTS = 5
LOCK_SECONDS = 5 * 60
SESSION_SECONDS = 8 * 60 * 60
SESSION_KEY = 'phien_tai_khoan'
REGISTER_KEY = 'hien_dang_ky'
USERNAME = re.compile(r'[a-z0-9][a-z0-9_.-]{2,31}')


def _ma_hoa_mat_khau(password, salt):
    # Cấu hình scrypt 32 MiB theo OWASP, mỗi tài khoản dùng một salt riêng.
    return scrypt(password.encode('utf-8'), salt=salt, n=2**15, r=8, p=3,
                  dklen=32, maxmem=64 * 1024**2)


def _duong_dan_db():
    custom_path = os.environ.get('NHA_HANG_DB_TAI_KHOAN')
    if custom_path:
        return Path(custom_path).expanduser()
    return Path.home() / '.canh_bao_nha_hang' / 'tai_khoan.sqlite3'


def _mo_db(path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return sqlite3.connect(path, timeout=10)


def _tao_bang(db):
    db.execute("""
        CREATE TABLE IF NOT EXISTS tai_khoan (
            ten TEXT PRIMARY KEY,
            salt BLOB NOT NULL,
            mat_khau_bam BLOB NOT NULL,
            so_lan_thu INTEGER NOT NULL DEFAULT 0,
            khoa_den INTEGER NOT NULL DEFAULT 0
        )
    """)


def _dang_ky(path, name, password):
    salt = token_bytes(16)
    digest = _ma_hoa_mat_khau(password, salt)
    with closing(_mo_db(path)) as db, db:
        _tao_bang(db)
        cursor = db.execute("""
            INSERT INTO tai_khoan (ten, salt, mat_khau_bam) VALUES (?, ?, ?)
            ON CONFLICT(ten) DO NOTHING
        """, (name, salt, digest))
        return cursor.rowcount == 1


def _dang_nhap(path, name, password):
    now = int(time())
    with closing(_mo_db(path)) as db:
        # Ghi nhận lượt thử trước khi kiểm tra mật khẩu để giới hạn cả yêu cầu đồng thời.
        with db:
            _tao_bang(db)
            cursor = db.execute("""
                UPDATE tai_khoan
                SET so_lan_thu = CASE WHEN khoa_den > 0 THEN 1 ELSE so_lan_thu + 1 END,
                    khoa_den = CASE
                        WHEN khoa_den = 0 AND so_lan_thu + 1 >= ? THEN ?
                        ELSE 0
                    END
                WHERE ten = ? AND khoa_den <= ?
            """, (MAX_ATTEMPTS, now + LOCK_SECONDS, name, now))
            row = db.execute('SELECT salt, mat_khau_bam FROM tai_khoan WHERE ten = ?',
                             (name,)).fetchone() if cursor.rowcount else None

        salt, expected = row if row else (b'\x00' * 16, b'\x00' * 32)
        matches = compare_digest(_ma_hoa_mat_khau(password, salt), expected)
        if row and matches:
            with db:
                db.execute('UPDATE tai_khoan SET so_lan_thu = 0, khoa_den = 0 WHERE ten = ?', (name,))
            return True
        return False


def _ma_dang_ky():
    setting = os.environ.get('NHA_HANG_MA_DANG_KY', '')
    try:
        return str(st.secrets.get('NHA_HANG_MA_DANG_KY', setting))
    except FileNotFoundError:
        return setting


def _vao_ung_dung(name):
    st.session_state.pop(REGISTER_KEY, None)
    st.session_state[SESSION_KEY] = (name, time() + SESSION_SECONDS)
    st.rerun()


def _form_dang_nhap(path):
    with st.form('form_dang_nhap', clear_on_submit=True, border=False):
        name = st.text_input('Tên đăng nhập', key='ten_dang_nhap', max_chars=32)
        password = st.text_input('Mật khẩu', key='mat_khau_dang_nhap', type='password',
                                 max_chars=MAX_PASSWORD)
        submitted = st.form_submit_button('Đăng nhập', type='primary')

    if submitted:
        name = name.strip().lower()
        if USERNAME.fullmatch(name) and MIN_PASSWORD <= len(password) <= MAX_PASSWORD and _dang_nhap(path, name, password):
            _vao_ung_dung(name)
        st.error('Kiểm tra tên đăng nhập và mật khẩu. Nếu đã thử nhiều lần, hãy đợi 5 phút.')


def _form_dang_ky(path, invite):
    with st.form('form_dang_ky', clear_on_submit=True, border=False):
        name = st.text_input('Tên đăng nhập', key='ten_dang_ky', max_chars=32,
                             placeholder='Ví dụ dat123')
        password = st.text_input('Mật khẩu', key='mat_khau_dang_ky', type='password',
                                 max_chars=MAX_PASSWORD, placeholder=f'Từ {MIN_PASSWORD} ký tự')
        code = st.text_input('Mã mời', type='password', max_chars=256) if invite else ''
        submitted = st.form_submit_button('Tạo tài khoản', type='primary')

    if submitted:
        name = name.strip().lower()
        if invite and not compare_digest(code.encode('utf-8'), invite.encode('utf-8')):
            st.error('Mã mời không đúng.')
        elif not USERNAME.fullmatch(name):
            st.error('Chọn tên từ 3 đến 32 ký tự, bắt đầu bằng chữ không dấu hoặc số. '
                     'Bạn có thể dùng thêm dấu chấm, gạch dưới hoặc gạch ngang.')
        elif not MIN_PASSWORD <= len(password) <= MAX_PASSWORD:
            st.error(f'Mật khẩu cần từ {MIN_PASSWORD} ký tự.')
        elif _dang_ky(path, name, password):
            _vao_ung_dung(name)
        else:
            st.error('Tên đăng nhập này đã được sử dụng.')


def yeu_cau_dang_nhap():
    session = st.session_state.get(SESSION_KEY)
    if session and session[1] > time():
        st.sidebar.caption(f'Tài khoản · {session[0]}')
        if st.sidebar.button('Đăng xuất'):
            st.session_state.clear()
            st.rerun()
        return

    st.session_state.pop(SESSION_KEY, None)
    register = st.session_state.get(REGISTER_KEY, False)
    _, center, _ = st.columns([1, 1.3, 1])
    with center.container(border=True):
        st.subheader('Tạo tài khoản' if register else 'Đăng nhập')
        if session:
            st.info('Phiên đã hết hạn. Vui lòng đăng nhập lại.')
        path = _duong_dan_db()
        try:
            if register:
                _form_dang_ky(path, _ma_dang_ky())
            else:
                _form_dang_nhap(path)
        except (OSError, sqlite3.Error):
            st.error('Không thể truy cập dữ liệu tài khoản trên máy chủ.')
        st.button('Quay lại đăng nhập' if register else 'Tạo tài khoản',
                  key='doi_form_tai_khoan',
                  on_click=st.session_state.update, kwargs={REGISTER_KEY: not register})
    st.stop()
