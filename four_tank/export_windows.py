"""Export the encrypted residual ciphertexts of the paper's example (linear plant, v = 64, q = 16381, N = 192
samples per key epoch) so that the heavy lattice reduction can run separately, for example overnight.

    python export_windows.py            ->  windows_linear_paper.npz   (P: 32 x 192 x 64, b: 32 x 192, + metadata)
"""
import numpy as np
import experiment as ex


def main(out="windows_linear_paper.npz"):
    sc = ex.default_scenario("linear")
    log = ex.run("linear", "encrypted", sc, keep_ciphertexts=True)
    T = log["lwe"]["T"]
    Pw = log["res_ct"][:, :-1].reshape(-1, T, log["lwe"]["v"])
    bw = log["res_ct"][:, -1].reshape(-1, T)
    np.savez(out, P=Pw, b=bw, q=log["q"], T=T, v=log["lwe"]["v"], sigma=log["lwe"]["sigma"], r=log["lwe"]["r"],
             sensor_window=log["sensor_window"], actuator_window=log["actuator_window"],
             dec_errors=log["dec_errors"], rho=log["rho"])
    print(f"wrote {out}: {Pw.shape[0]} windows of {T} residual ciphertexts, decryption errors in the run: {log['dec_errors']}")


if __name__ == "__main__":
    main()
