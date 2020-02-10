## WebCrawling

크롬에서 `ctrl`+`shift`+`I`: 개발자 도구



find_all(태그이름, '키':'value)

top_keywords = soup.find('span', {'class':'tit'})[:20]



#### HTML 구조

table class='tbl_home'

```html
<table>
 <thead>
	<tr>  table row
		<th> 1행 1열 </th>
		<th> 1행 2열 </th>
    </tr>
 </thead>
 <tbody>
	<tr>  table row
		<td> 2행 1열 </td>
		<td> 2행 2열 </td>
	</tr>
 </tbody>
</table>
```



#### 공공데이터 데이터 이용

[공공데이터](https://www.data.go.kr/subMain.jsp#/L3B1YnIvcG90L215cC9Jcm9zTXlQYWdlL29wZW5EZXZEZXRhaWxQYWdlJEBeMDgyTTAwMDAxMzBeTTAwMDAxMzUkQF5wdWJsaWNEYXRhRGV0YWlsUGs9dWRkaTo0MWM5YTBlMy02YzgwLTQ3YjMtODhjMC04NDk4YzgwYmMyZWMkQF5wcmN1c2VSZXFzdFNlcU5vPTEwODMwODc1JEBecmVxc3RTdGVwQ29kZT1TVENEMDE=) 접속 사이트

HTTP 요청 방식: **REST**

일반 인증키: 인증키

End Point: API 시작지점

참고문서를 사용하여 End Point 뒤에 query 등