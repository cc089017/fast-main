import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

const http = axios.create({
  baseURL: API_BASE,
  withCredentials: true, // ★ 쿠키 전송
  headers: {
    "Content-Type": "application/json",
  },
});

export default http;
