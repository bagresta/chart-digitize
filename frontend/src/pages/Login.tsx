import { type FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { login } from "../api";

export function Login() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const navigate = useNavigate();

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(password);
      navigate("/upload");
    } catch {
      setError("Incorrect password");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} style={{ maxWidth: 320, margin: "80px auto" }}>
      <h1>Chart Digitizer</h1>
      <input
        type="password"
        value={password}
        onChange={(event) => setPassword(event.target.value)}
        placeholder="Password"
        style={{ width: "100%", padding: 8 }}
      />
      {error && <p style={{ color: "red" }}>{error}</p>}
      <button type="submit" disabled={submitting} style={{ marginTop: 12, width: "100%", padding: 8 }}>
        {submitting ? "Signing in..." : "Sign in"}
      </button>
    </form>
  );
}
