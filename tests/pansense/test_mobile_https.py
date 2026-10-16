from cryptography import x509
from cryptography.hazmat.primitives import serialization

from pansense.mobile_https import certificates,lan_address


def test_certificate_has_lan_ip_and_reuses_ca_without_installing_it(tmp_path):
    first=certificates(tmp_path,'192.168.1.25')
    ca_bytes=(tmp_path/'ca.pem').read_bytes()
    leaf=x509.load_pem_x509_certificate(first['cert'].read_bytes())
    assert '192.168.1.25' in str(leaf.extensions.get_extension_for_class(x509.SubjectAlternativeName).value)
    ca=x509.load_der_x509_certificate(first['ca'].read_bytes())
    assert ca.subject==leaf.issuer
    second=certificates(tmp_path,'192.168.1.26')
    assert (tmp_path/'ca.pem').read_bytes()==ca_bytes
    assert first['fingerprint']==second['fingerprint']
    assert second['key'].read_bytes().startswith(b'-----BEGIN PRIVATE KEY-----')


def test_wifi_address_selection_ignores_tun_benchmark_and_loopback(monkeypatch):
    monkeypatch.setattr('pansense.mobile_https.socket.gethostbyname_ex',lambda *args:('host',[],['198.18.0.1','127.0.0.1','10.20.234.18']))
    assert lan_address()=='10.20.234.18'
