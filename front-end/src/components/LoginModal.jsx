import React, { useRef, useEffect, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

function LoginModal({ onClose, onSignupOpen, onLoginSuccess }) {
  const modalRef = useRef(null);
  const [userId, setUserId] = useState("");   // ✅ email 대신 userId
  const [pw, setPw] = useState("");
  const [loading, setLoading] = useState(false);

  // 바깥 클릭 닫기
  useEffect(() => {
    const handleOutsideClick = (e) => {
      if (modalRef.current && !modalRef.current.contains(e.target)) onClose();
    };
    document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, [onClose]);

  const handleSignup = () => {
    onClose();
    onSignupOpen && onSignupOpen();
  };

  const handleLogin = async () => {
    if (!userId || !pw) {
      alert("아이디/비밀번호를 입력해 주세요.");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: userId, password: pw }), // ✅ email → id
      });

      const data = await res.json();
      if (!res.ok) {
        const msg =
          data?.message ||
          (Array.isArray(data?.detail)
            ? data.detail.map((d) => d.msg).join("\n")
            : data?.detail || "로그인에 실패했습니다.");
        throw new Error(msg);
      }

      localStorage.setItem("access_token", data.access_token);
      onLoginSuccess && onLoginSuccess(data);
      alert("로그인 성공!");
      onClose();
    } catch (err) {
      alert(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
      <div ref={modalRef} className="bg-white p-6 rounded-lg w-[30rem] shadow-lg">
        <h2 className="text-4xl font-normal mb-12 text-center">로그인</h2>

        {/* ✅ 아이디 입력 */}
        <input
          type="text"
          placeholder="아이디"
          className="w-full text-lg mb-3 px-4 py-2 border rounded"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          autoComplete="username"
        />

        {/* 비밀번호 입력 */}
        <input
          type="password"
          placeholder="비밀번호"
          className="w-full text-lg mb-4 px-4 py-2 border rounded"
          value={pw}
          onChange={(e) => setPw(e.target.value)}
          autoComplete="current-password"
        />

        <button
          onClick={handleLogin}
          disabled={loading}
          className="w-full text-xl bg-blue-600 text-white mt-2 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-60"
        >
          {loading ? "로그인 중..." : "로그인"}
        </button>

        <button
          onClick={handleSignup}
          className="w-full text-sm text-gray-500 hover:text-blue-600 mt-6 pt-5 border-t"
        >
          회원가입
        </button>
      </div>
    </div>
  );
}

export default LoginModal;
