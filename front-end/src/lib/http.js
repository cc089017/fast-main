import axios from "axios";

// ✅ baseURL 비움(상대경로). 모든 호출은 "/api/..." 로만!
const http = axios.create({
  baseURL: "",
  withCredentials: true, // 쿠키 포함
  headers: { "Content-Type": "application/json" },
});

export default http;
